import numpy as np


# Exact solved source pairs, including their colors and dimensions. Reusing a
# small motif under a new palette or in a larger canvas is allowed.
_SOURCE_PAIRS = frozenset({
    ((0, 0, 0, 2, 0, 2, 0, 0, 0), 1),
    ((2, 2, 0, 0, 2, 2, 0, 0, 0), 7),
    ((2, 2, 0, 0, 2, 0, 0, 0, 0), 7),
    ((2, 0, 0, 2, 0, 0, 0, 2, 0), 7),
    ((2, 0, 2, 2, 0, 2, 2, 0, 2), 1),
    ((2, 0, 2, 0, 2, 0, 2, 0, 2), 1),
    ((0, 0, 0, 2, 0, 0, 2, 0, 0), 7),
    ((2, 0, 2, 2, 2, 2, 2, 0, 2), 1),
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, seed: int,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
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


def _key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _signature(puzzle: dict) -> tuple:
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[part]) for part in ("train", "test"))


def _sym(grid: np.ndarray, axis: int) -> bool:
    return bool(np.array_equal(grid, np.flip(grid, axis=axis)))


def _other(axis: int) -> int:
    return 1 - axis


def _diagonal(grid: np.ndarray, anti: bool = False) -> bool:
    if grid.shape[0] != grid.shape[1]:
        return False
    reflected = grid[::-1, ::-1].T if anti else grid.T
    return bool(np.array_equal(grid, reflected))


def _hypotheses(grid: np.ndarray, axis: int) -> tuple[bool, ...]:
    target = _sym(grid, axis)
    other = _sym(grid, _other(axis))
    rotated = bool(np.array_equal(grid, grid[::-1, ::-1]))
    main = _diagonal(grid)
    anti = _diagonal(grid, True)
    if axis == 1:
        left = grid[:, :grid.shape[1] // 2]
        right = grid[:, (grid.shape[1] + 1) // 2:]
    else:
        left = grid[:grid.shape[0] // 2, :]
        right = grid[(grid.shape[0] + 1) // 2:, :]
    equal_sides = int(np.count_nonzero(left == grid.flat[0])) == int(
        np.count_nonzero(right == grid.flat[0]))
    return (target, other, rotated, main, anti, target or other,
            target or main, target or anti, target or rotated,
            target or other or main or anti, target and other, equal_sides)


def _possible_predictions(train: list[dict], grid: np.ndarray, axis: int) -> set[int]:
    values = [_hypotheses(pair["input"], axis) for pair in train]
    labels = [int(pair["output"][0, 0]) for pair in train]
    predicted = set()
    for index in range(len(values[0])):
        mapping = {}
        for features, label in zip(values, labels):
            state = features[index]
            if state in mapping and mapping[state] != label:
                break
            mapping[state] = label
        else:
            state = _hypotheses(grid, axis)[index]
            if state in mapping:
                predicted.add(mapping[state])
    return predicted


def _witness_pool(axis: int, size: tuple[int, int]) -> tuple[dict[int, list[np.ndarray]],
                                                            dict[int, list[np.ndarray]],
                                                            dict[int, list[np.ndarray]]]:
    # A small finite pool is used only for the teaching witnesses. Other pairs
    # are composed at arbitrary sizes up to 30 by 30.
    positive: dict[int, list[np.ndarray]] = {}
    negative: dict[int, list[np.ndarray]] = {}
    other_only: dict[int, list[np.ndarray]] = {}
    cells = size[0] * size[1]
    for code in range(1, (1 << cells) - 1):
        mask = np.asarray([(code >> bit) & 1 for bit in range(cells)],
                          dtype=np.int8).reshape(size)
        target = _sym(mask, axis)
        other = _sym(mask, _other(axis))
        rotated = bool(np.array_equal(mask, mask[::-1, ::-1]))
        diagonal = _diagonal(mask) or _diagonal(mask, True)
        amount = int(mask.sum())
        if target and not (other or rotated or diagonal):
            positive.setdefault(amount, []).append(mask)
        elif not target and not (other or rotated or diagonal):
            negative.setdefault(amount, []).append(mask)
        elif other and not target and not (rotated or diagonal):
            other_only.setdefault(amount, []).append(mask)
    return positive, negative, other_only


def _teaching_masks(rng: np.random.Generator, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_size = (3, 3) if int(rng.integers(0, 3)) == 0 else (4, 3)
    size = base_size if axis == 1 else base_size[::-1]
    positive, negative, other_only = _witness_pool(axis, size)
    shared = sorted(set(positive) & set(negative))
    amount = shared[int(rng.integers(len(shared)))]
    pos = positive[amount][int(rng.integers(len(positive[amount])))].copy()
    neg = negative[amount][int(rng.integers(len(negative[amount])))].copy()
    choices = [mask for masks in other_only.values() for mask in masks]
    opposite = choices[int(rng.integers(len(choices)))].copy()
    return pos, neg, opposite


def _shape(rng: np.random.Generator, regime: int, section: str) -> tuple[int, int]:
    if regime == 0:
        return (3, 3) if rng.random() < 0.65 else (
            int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    if rng.random() < 0.18:
        return (3, 3)
    if regime == 1:
        ceiling = 12 if section == "train" else 16
    else:
        ceiling = 16 if section == "train" else 30
    floor = 3 if section == "train" or rng.random() < 0.35 else 6
    return int(rng.integers(floor, ceiling + 1)), int(rng.integers(floor, ceiling + 1))


def _mirror_pairs(shape: tuple[int, int], axis: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    height, width = shape
    if axis == 1:
        return [((r, c), (r, width - 1 - c))
                for r in range(height) for c in range(width // 2)]
    return [((r, c), (height - 1 - r, c))
            for r in range(height // 2) for c in range(width)]


def _random_mask(rng: np.random.Generator, shape: tuple[int, int],
                 axis: int, symmetric: bool, style: int) -> np.ndarray:
    pairs = _mirror_pairs(shape, axis)
    for _ in range(32):
        mask = np.zeros(shape, dtype=np.int8)
        density = float(rng.choice(np.asarray([.06, .15, .28, .42, .58, .72, .86, .94])))
        for first, second in pairs:
            value = int(rng.random() < density)
            mask[first] = value
            mask[second] = value
        if axis == 1 and shape[1] % 2:
            mask[:, shape[1] // 2] = (rng.random(shape[0]) < density).astype(np.int8)
        elif axis == 0 and shape[0] % 2:
            mask[shape[0] // 2, :] = (rng.random(shape[1]) < density).astype(np.int8)
        if not symmetric:
            if style == 2 and len(pairs) >= 2:
                zeros = [i for i, (a, b) in enumerate(pairs) if mask[a] == 0]
                ones = [i for i, (a, b) in enumerate(pairs) if mask[a] == 1]
                if not zeros or not ones:
                    continue
                first = pairs[zeros[int(rng.integers(len(zeros)))]][0]
                second = pairs[ones[int(rng.integers(len(ones)))]][0]
                mask[first] = 1
                mask[second] = 0
            elif style == 3 and min(shape) >= 4:
                mask[:] = 0
                if axis == 1:
                    column = int(rng.choice(np.asarray([0, shape[1] - 1])))
                    start = int(rng.integers(0, shape[0] - 1))
                    mask[start:start + 2, column] = 1
                else:
                    row = int(rng.choice(np.asarray([0, shape[0] - 1])))
                    start = int(rng.integers(0, shape[1] - 1))
                    mask[row, start:start + 2] = 1
            else:
                maximum = min(len(pairs), 8)
                mismatch_count = 1 if style == 0 else int(rng.integers(2, maximum + 1))
                indices = rng.choice(len(pairs), size=mismatch_count, replace=False)
                for index in indices:
                    first, second = pairs[int(index)]
                    selected = first if rng.random() < .5 else second
                    mask[selected] = 1 - mask[selected]
        if 0 < int(mask.sum()) < mask.size and _sym(mask, axis) == symmetric:
            return mask
    # A deterministic legal fallback for extremely sparse small canvases.
    mask = np.zeros(shape, dtype=np.int8)
    if symmetric:
        if axis == 1:
            mask[0, 0] = mask[0, -1] = 1
        else:
            mask[0, 0] = mask[-1, 0] = 1
    else:
        mask[0, 0] = 1
    return mask


def _small_mask(rng: np.random.Generator, axis: int,
                symmetric: bool, amount: int) -> np.ndarray:
    choices = []
    for code in range(1, 511):
        mask = np.asarray([(code >> bit) & 1 for bit in range(9)],
                          dtype=np.int8).reshape(3, 3)
        if int(mask.sum()) == amount and _sym(mask, axis) == symmetric:
            choices.append(mask)
    return choices[int(rng.integers(len(choices)))].copy()


def _pair(mask: np.ndarray, palette: tuple[int, int], labels: tuple[int, int],
          axis: int) -> dict:
    grid = np.where(mask == 1, palette[1], palette[0]).astype(np.int8)
    label = labels[0] if _sym(grid, axis) else labels[1]
    return {"input": grid, "output": np.asarray([[label]], dtype=np.int8)}


def _palette(rng: np.random.Generator, labels: tuple[int, int]) -> tuple[int, int]:
    available = [color for color in range(10) if color not in labels]
    chosen = rng.choice(available, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _is_source(pair: dict) -> bool:
    grid = pair["input"]
    return grid.shape == (3, 3) and (
        tuple(int(x) for x in grid.flat), int(pair["output"][0, 0])) in _SOURCE_PAIRS


def _add(pair: dict, section: list[dict], used: set[tuple]) -> bool:
    key = _key(pair["input"])
    if key in used or _is_source(pair):
        return False
    used.add(key)
    section.append(pair)
    return True


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    axis = int(rng.integers(0, 2))
    chosen = rng.choice(10, size=2, replace=False)
    labels = int(chosen[0]), int(chosen[1])
    regime = int(rng.choice(np.asarray([0, 1, 2]), p=[.25, .45, .30]))
    used: set[tuple] = set()
    train: list[dict] = []
    test: list[dict] = []

    positive, negative, other_only = _teaching_masks(rng, axis)
    for mask in (positive, negative):
        for _ in range(40):
            pair = _pair(mask, _palette(rng, labels), labels, axis)
            if train and tuple(sorted(int(x) for x in np.unique(pair["input"]))) == (
                tuple(sorted(int(x) for x in np.unique(train[0]["input"])))
            ):
                continue
            if _add(pair, train, used):
                break
        else:
            raise AssertionError("unable to place teaching pair")
    if train_count >= 3:
        for _ in range(40):
            pair = _pair(other_only, _palette(rng, labels), labels, axis)
            if _add(pair, train, used):
                break
        else:
            raise AssertionError("unable to place alternative-axis witness")

    # Sometimes demonstrate that changing only the self-reflecting center
    # leaves a positive label unchanged.
    if train_count >= 4 and rng.random() < .55:
        center = positive.copy()
        if axis == 1:
            center[:, center.shape[1] // 2] = 1 - center[:, center.shape[1] // 2]
        else:
            center[center.shape[0] // 2, :] = 1 - center[center.shape[0] // 2, :]
        bg = int(train[0]["input"][tuple(np.argwhere(positive == 0)[0])])
        fg = int(train[0]["input"][tuple(np.argwhere(positive == 1)[0])])
        if 0 < int(center.sum()) < center.size:
            _add(_pair(center, (bg, fg), labels, axis), train, used)

    for index in range(len(train), train_count):
        for attempt in range(100):
            symmetric = bool((index + seed) % 2)
            shape = _shape(rng, regime, "train")
            style = int(rng.choice(4, p=[.38, .35, .19, .08]))
            mask = _random_mask(rng, shape, axis, symmetric, style)
            pair = _pair(mask, _palette(rng, labels), labels, axis)
            if _add(pair, train, used):
                break
        else:
            raise AssertionError("unable to place training pair")

    # Tests are generated after the teaching set. A held-out pattern is used
    # only when reasonable symmetry alternatives surviving training agree.
    extreme_shift = regime == 0 and test_count >= 2 and rng.random() < .8
    original_extremes = bool(rng.integers(0, 2))
    for index in range(test_count):
        for attempt in range(300):
            symmetric = bool((index + seed // 3) % 2) if test_count > 1 else bool(rng.integers(0, 2))
            shape = _shape(rng, regime, "test")
            if regime == 0 and rng.random() < .30:
                shape = (3, 3)
            style = int(rng.choice(4, p=[.30, .34, .22, .14]))
            if extreme_shift and index < 2:
                symmetric = index == 1
                if original_extremes:
                    amount = 7 if symmetric else 2
                else:
                    amount = 1 if symmetric else 7
                mask = _small_mask(rng, axis, symmetric, amount)
            else:
                mask = _random_mask(rng, shape, axis, symmetric, style)
            pair = _pair(mask, _palette(rng, labels), labels, axis)
            if _key(pair["input"]) in used or _is_source(pair):
                continue
            predicted = _possible_predictions(train, pair["input"], axis)
            if predicted == {int(pair["output"][0, 0])}:
                _add(pair, test, used)
                break
        else:
            raise AssertionError("unable to place learnable test pair")

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "sections must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    inputs: set[tuple] = set()
    pairs: set[tuple] = set()
    for part in (train, test):
        for pair in part:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must contain exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grids must be integer")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "illegal dimensions")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "illegal color")
            grid, output = pair["input"], pair["output"]
            _require(grid.size > 1 and len(np.unique(grid)) == 2, "input must be binary")
            _require(output.shape == (1, 1), "output must be one cell")
            _require(int(output[0, 0]) not in set(int(x) for x in np.unique(grid)),
                     "input and output colors must differ")
            key = _key(grid)
            pair_key = (key, _key(output))
            _require(key not in inputs and pair_key not in pairs, "duplicate pair")
            _require(not _is_source(pair), "exact source pair copied")
            inputs.add(key)
            pairs.add(pair_key)

    labels = {int(pair["output"][0, 0]) for pair in train}
    _require(len(labels) == 2, "training must demonstrate both labels")
    _require(all(int(pair["output"][0, 0]) in labels for pair in test),
             "test uses an unseen label")
    viable: list[tuple[int, dict[bool, int]]] = []
    for axis in (0, 1):
        mapping: dict[bool, int] = {}
        for pair in train:
            state = _sym(pair["input"], axis)
            label = int(pair["output"][0, 0])
            if state in mapping and mapping[state] != label:
                break
            mapping[state] = label
        else:
            if len(mapping) == 2 and mapping[True] != mapping[False]:
                viable.append((axis, mapping))
    _require(len(viable) == 1, "training does not determine a unique axis and label mapping")
    axis, mapping = viable[0]
    _require(any(_sym(pair["input"], axis) for pair in train) and
             any(not _sym(pair["input"], axis) for pair in train),
             "training lacks a class")
    for pair in test:
        _require(mapping[_sym(pair["input"], axis)] == int(pair["output"][0, 0]),
                 "test output contradicts the training rule")
        _require(_possible_predictions(train, pair["input"], axis)
                 == {int(pair["output"][0, 0])},
                 "test is ambiguous under reasonable symmetry alternatives")


def _expect_value_error(seed: int, train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(seed, train, test)
    except ValueError:
        return
    raise AssertionError("illegal counts were accepted")


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    boundary = [generate_puzzle(8001, 2, 1), generate_puzzle(8002, 10, 1),
                generate_puzzle(8003, 10, 4)]
    partial = [generate_puzzle(8101, None, 1), generate_puzzle(8102, None, 4),
               generate_puzzle(8103, 2, None), generate_puzzle(8104, 10, None)]
    for puzzle in automatic + boundary + partial:
        validate_generated_puzzle(puzzle)
    _require(_signature(generate_puzzle(7)) == _signature(generate_puzzle(7)),
             "generation is not deterministic")
    _require(len({_signature(p) for p in automatic}) >= 40,
             "insufficient cross-seed diversity")
    _require(len({len(p["train"]) for p in automatic}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in automatic}) >= 3, "test counts lack range")
    _require([len(p["train"]) for p in boundary] == [2, 10, 10],
             "boundary train counts changed")
    _require([len(p["test"]) for p in boundary] == [1, 1, 4],
             "boundary test counts changed")
    _require([len(partial[0]["test"]), len(partial[1]["test"])] == [1, 4],
             "partial test counts changed")
    _require([len(partial[2]["train"]), len(partial[3]["train"])] == [2, 10],
             "partial train counts changed")
    for index, counts in enumerate([(1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4), (2, 4)]):
        _expect_value_error(8200 + index, *counts)
    dimensions = set()
    densities = {True: set(), False: set()}
    axes = set()
    palettes = set()
    mismatch_counts = set()
    equal_side_negatives = 0
    center_only_puzzles = 0
    source_shift_puzzles = 0
    rectangular = 0
    even_width = 0
    large = 0
    for puzzle in automatic:
        first = puzzle["train"][0]
        second = puzzle["train"][1]
        _require(first["input"].shape == second["input"].shape,
                 "teaching witnesses have unrelated shapes")
        _require(sorted(np.unique(first["input"], return_counts=True)[1]) ==
                 sorted(np.unique(second["input"], return_counts=True)[1]),
                 "teaching witnesses leak their labels through density")
        axis = next(axis for axis in (0, 1) if all(
            _sym(pair["input"], axis) == (int(pair["output"][0, 0]) ==
              int(first["output"][0, 0])) for pair in puzzle["train"]))
        axes.add(axis)
        _require(len({tuple(sorted(int(x) for x in np.unique(p["input"])))
                      for p in puzzle["train"]}) >= 2,
                 "training input colors do not vary")
        for i, pair in enumerate(puzzle["train"]):
            a = pair["input"]
            if not _sym(a, axis):
                continue
            for later in puzzle["train"][i + 1:]:
                b = later["input"]
                if not _sym(b, axis) or a.shape != b.shape or np.array_equal(a, b):
                    continue
                if axis == 1 and a.shape[1] % 2:
                    same_sides = np.array_equal(
                        np.delete(a, a.shape[1] // 2, axis=1),
                        np.delete(b, b.shape[1] // 2, axis=1))
                elif axis == 0 and a.shape[0] % 2:
                    same_sides = np.array_equal(
                        np.delete(a, a.shape[0] // 2, axis=0),
                        np.delete(b, b.shape[0] // 2, axis=0))
                else:
                    same_sides = False
                if same_sides:
                    center_only_puzzles += 1
                    break
            else:
                continue
            break
        positive_two = False
        negative_two = False
        for part in ("train", "test"):
            for pair in puzzle[part]:
                grid = pair["input"]
                state = _sym(grid, axis)
                dimensions.add(grid.shape)
                palettes.add(tuple(sorted(int(x) for x in np.unique(grid))))
                rectangular += grid.shape[0] != grid.shape[1]
                even_width += grid.shape[1] % 2 == 0
                large += max(grid.shape) >= 20
                density = min(int(np.count_nonzero(grid == grid.flat[0])),
                              int(np.count_nonzero(grid != grid.flat[0])))
                densities[state].add(density)
                if part == "test" and grid.shape == (3, 3) and density == 2:
                    if state:
                        positive_two = True
                    else:
                        negative_two = True
                if not state:
                    mismatch_counts.add(sum(grid[a] != grid[b]
                        for a, b in _mirror_pairs(grid.shape, axis)))
                    equal_side_negatives += _hypotheses(grid, axis)[-1]
        source_shift_puzzles += positive_two and negative_two
    _require(axes == {0, 1}, "both axis variants were not exercised")
    _require(len(dimensions) >= 30 and any(max(s) >= 20 for s in dimensions),
             "grid dimensions lack diversity")
    _require(len(palettes) >= 12, "pair palettes lack diversity")
    _require(len(densities[True] & densities[False]) >= 4,
             "symmetric and asymmetric densities do not overlap")
    _require(1 in mismatch_counts and any(x >= 3 for x in mismatch_counts),
             "mismatch counts lack diversity")
    _require(equal_side_negatives >= 10, "equal-side-count negatives are rare")
    _require(center_only_puzzles >= 5, "center-column demonstrations are rare")
    _require(source_shift_puzzles >= 1, "source extreme test shift is absent")
    _require(rectangular >= 30 and even_width >= 30 and large >= 5,
             "geometry regimes lack diversity")
