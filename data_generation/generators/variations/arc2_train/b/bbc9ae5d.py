"""Generate variations of the left-anchored growing-prefix ARC task."""

import numpy as np


_WIDTHS = tuple(range(6, 31, 2))
# The six literal input signatures in task.json: (width, background, foreground, K).
_SOURCE_SIGNATURES = frozenset({
    (6, 0, 8, 4), (8, 0, 2, 1), (10, 0, 5, 3),
    (6, 0, 7, 1), (6, 0, 1, 2), (12, 0, 1, 3),
})


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    if num_train is not None:
        if not _is_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choice(rng: np.random.Generator, values) -> int:
    values = tuple(values)
    if not values:
        raise ValueError("empty sampling pool")
    return int(values[int(rng.integers(len(values)))])


def _signature(width: int, background: int, foreground: int, prefix: int) -> tuple[int, int, int, int]:
    return width, background, foreground, prefix


def _available_roles(
    width: int,
    prefix: int,
    used: set[tuple[int, int, int, int]],
    backgrounds=range(10),
    foregrounds=range(10),
) -> list[tuple[int, int]]:
    return [
        (background, foreground)
        for background in backgrounds
        for foreground in foregrounds
        if background != foreground
        and _signature(width, background, foreground, prefix) not in used
        and _signature(width, background, foreground, prefix) not in _SOURCE_SIGNATURES
    ]


def _choose_roles(
    rng: np.random.Generator,
    width: int,
    prefix: int,
    used: set[tuple[int, int, int, int]],
    backgrounds=range(10),
    foregrounds=range(10),
) -> tuple[int, int]:
    roles = _available_roles(width, prefix, used, backgrounds, foregrounds)
    if not roles:
        raise AssertionError("no distinct unused color roles available")
    return roles[int(rng.integers(len(roles)))]


def _make_pair(width: int, background: int, foreground: int, prefix: int) -> dict:
    if width not in _WIDTHS or not 1 <= prefix <= width // 2 + 1:
        raise ValueError("invalid staircase geometry")
    if not (0 <= background <= 9 and 0 <= foreground <= 9) or background == foreground:
        raise ValueError("foreground and background must be distinct ARC colors")
    input_grid = np.full((1, width), background, dtype=np.int8)
    input_grid[0, :prefix] = foreground
    output_grid = np.full((width // 2, width), background, dtype=np.int8)
    for row in range(width // 2):
        output_grid[row, :prefix + row] = foreground
    return {"input": input_grid, "output": output_grid}


def _analyze_input(grid: np.ndarray) -> tuple[int, int, int, int]:
    _require(grid.shape[0] == 1, "input must have one row")
    width = int(grid.shape[1])
    _require(width in _WIDTHS, "input width must be even and between 6 and 30")
    row = grid[0]
    foreground = int(row[0])
    changes = np.flatnonzero(row != foreground)
    _require(len(changes) > 0, "input needs a visible background suffix")
    prefix = int(changes[0])
    background = int(row[prefix])
    _require(bool(np.all(row[prefix:] == background)), "input has gaps or extra colors")
    _require(1 <= prefix <= width // 2 + 1, "prefix exceeds staircase capacity")
    return width, background, foreground, prefix


def _sample_train_geometry(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    # Give source-scale training (6 and 10) a recurring, but minority, share.
    high = _choice(rng, (10, 10, 10, *range(12, 31, 2)))
    low = _choice(rng, (width for width in _WIDTHS if width <= high - 4))
    base = _choice(rng, range(1, low // 2 + 2))
    alternate = _choice(rng, (k for k in range(1, low // 2 + 2) if k != base))
    if count == 2:
        return [(low, base), (high, alternate)]
    geometry = [(low, base), (low, alternate), (high, base)]
    for _ in range(count - 3):
        width = _choice(rng, (w for w in _WIDTHS if w <= high))
        geometry.append((width, _choice(rng, range(1, width // 2 + 2))))
    return geometry


def _test_width(rng: np.random.Generator, train_widths: set[int]) -> int:
    high = max(train_widths)
    category = _choice(rng, (0, 1, 2, 2))
    if category == 0:  # Width already demonstrated, with a new input grid.
        pool = train_widths
    elif category == 1:  # Smaller than the widest demonstration.
        pool = (width for width in _WIDTHS if width < high)
    else:  # Moderate or large extrapolation.
        pool = (width for width in _WIDTHS if width > high)
    options = sorted(pool)
    if not options:  # A width-30 training example leaves no larger ARC width.
        options = sorted(width for width in _WIDTHS if width < high)
    return _choice(rng, options)


def _test_prefix(rng: np.random.Generator, width: int, train_prefixes: set[int]) -> int:
    maximum = width // 2 + 1
    mode = _choice(rng, (0, 1, 2, 2, 3, 3, 4))
    pools = (
        (1,),
        (maximum,),
        tuple(k for k in range(1, maximum + 1) if k not in train_prefixes),
        tuple(k for k in range(1, maximum + 1) if k in train_prefixes),
        tuple(range(2, maximum)),
    )
    pool = pools[mode] or tuple(range(1, maximum + 1))
    return _choice(rng, pool)


def _test_roles(
    rng: np.random.Generator,
    width: int,
    prefix: int,
    used: set[tuple[int, int, int, int]],
    train_foregrounds: set[int],
) -> tuple[int, int]:
    mode = _choice(rng, (0, 0, 1, 1, 2))
    foregrounds = (
        train_foregrounds,
        set(range(10)) - train_foregrounds,
        {0},
    )[mode]
    if not foregrounds:
        foregrounds = set(range(10))
    background_mode = _choice(rng, (0, 0, 1, 1, 1))
    backgrounds = (0,) if background_mode == 0 else range(1, 10)
    options = _available_roles(width, prefix, used, backgrounds, sorted(foregrounds))
    if not options:
        options = _available_roles(width, prefix, used, range(10), sorted(foregrounds))
    if not options:
        options = _available_roles(width, prefix, used)
    return options[int(rng.integers(len(options)))]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Left anchoring, growth direction, and the width-to-height rule are fixed
    # conventions of this family. Color roles are visible in each input.
    geometry = _sample_train_geometry(rng, train_count)
    used: set[tuple[int, int, int, int]] = set()
    train_descriptors: list[tuple[int, int, int, int]] = []

    first_width, first_prefix = geometry[0]
    background, first_foreground = _choose_roles(
        rng, first_width, first_prefix, used, (0,), range(1, 10)
    )
    train_descriptors.append((first_width, background, first_foreground, first_prefix))
    used.add(train_descriptors[-1])

    second_width, second_prefix = geometry[1]
    if int(rng.integers(3)) == 0:
        second_foregrounds = (0,)
    else:
        second_foregrounds = tuple(c for c in range(1, 10) if c != first_foreground)
    background, foreground = _choose_roles(
        rng, second_width, second_prefix, used, range(1, 10), second_foregrounds
    )
    train_descriptors.append((second_width, background, foreground, second_prefix))
    used.add(train_descriptors[-1])

    if train_count >= 3:
        width, prefix = geometry[2]
        background, foreground = _choose_roles(
            rng, width, prefix, used, range(10), (first_foreground,)
        )
        train_descriptors.append((width, background, foreground, prefix))
        used.add(train_descriptors[-1])

    for width, prefix in geometry[3:]:
        known = {item[2] for item in train_descriptors}
        foregrounds = sorted(known) if int(rng.integers(2)) == 0 else range(10)
        backgrounds = (0,) if int(rng.integers(5)) < 2 else range(1, 10)
        options = _available_roles(width, prefix, used, backgrounds, foregrounds)
        if not options:
            options = _available_roles(width, prefix, used)
        background, foreground = options[int(rng.integers(len(options)))]
        descriptor = (width, background, foreground, prefix)
        train_descriptors.append(descriptor)
        used.add(descriptor)

    rng.shuffle(train_descriptors)
    train_pairs = [_make_pair(*descriptor) for descriptor in train_descriptors]
    train_widths = {item[0] for item in train_descriptors}
    train_prefixes = {item[3] for item in train_descriptors}
    train_foregrounds = {item[2] for item in train_descriptors}

    test_pairs = []
    for _ in range(test_count):
        width = _test_width(rng, train_widths)
        prefix = _test_prefix(rng, width, train_prefixes)
        background, foreground = _test_roles(rng, width, prefix, used, train_foregrounds)
        descriptor = (width, background, foreground, prefix)
        used.add(descriptor)
        test_pairs.append(_make_pair(*descriptor))

    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # Inference uses only demonstrations and the new input. In particular, the
    # test output and generation-time descriptors are unavailable here.
    ratios = set()
    steps = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        width, background, foreground, prefix = _analyze_input(inp)
        _require(np.array_equal(out[0], inp[0]), "training does not copy row zero")
        ratios.add((width, out.shape[0]))
        for row in range(1, out.shape[0]):
            previous = int(np.count_nonzero(out[row - 1] == foreground))
            current = int(np.count_nonzero(out[row] == foreground))
            steps.add(current - previous)
            _require(bool(np.all(out[row, current:] == background)), "training suffix changed")
        _require(int(np.count_nonzero(out[0] == foreground)) == prefix, "training prefix changed")
    _require(all(width == 2 * height for width, height in ratios), "training size rule unclear")
    _require(steps == {1}, "training growth rule unclear")
    width, background, foreground, prefix = _analyze_input(test_input)
    predicted = np.full((width // 2, width), background, dtype=np.int8)
    for row in range(width // 2):
        predicted[row, :prefix + row] = foreground
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "bad pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    descriptors: dict[str, list[tuple[int, int, int, int]]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{label}.{name} array")
                _require(all(1 <= size <= 30 for size in grid.shape), f"{label}.{name} shape")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label}.{name} dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label}.{name} colors")
            inp, out = pair["input"], pair["output"]
            width, background, foreground, prefix = _analyze_input(inp)
            descriptor = (width, background, foreground, prefix)
            descriptors[split].append(descriptor)
            _require(descriptor not in _SOURCE_SIGNATURES, f"{label} copies source")
            _require(out.shape == (width // 2, width), f"{label} output size")
            _require(np.array_equal(out, _make_pair(*descriptor)["output"]), f"{label} staircase rule")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, f"{label} duplicate")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    training = descriptors["train"]
    _require(len({item[0] for item in training}) >= 2, "training needs multiple widths")
    _require(len({item[3] for item in training}) >= 2, "training needs multiple initial prefixes")
    _require(len({item[2] for item in training}) >= 2, "training needs multiple foreground colors")
    _require(0 in {item[1] for item in training} and any(item[1] != 0 for item in training), "training needs both background types")
    if train_count >= 3:
        _require(any(a[0] == b[0] and a[3] != b[3] for i, a in enumerate(training) for b in training[i + 1:]), "same-width prefix variation absent")
        _require(any(a[0] != b[0] and a[3] == b[3] for i, a in enumerate(training) for b in training[i + 1:]), "same-prefix width variation absent")
        _require(any(a[0] != b[0] and a[2] == b[2] for i, a in enumerate(training) for b in training[i + 1:]), "foreground reuse absent")
    for pair in puzzle["test"]:
        predicted = _predict_from_training(puzzle["train"], pair["input"])
        _require(np.array_equal(predicted, pair["output"]), "test not recoverable from demonstrations")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"])) for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _require(len({_fingerprint(p) for p in puzzles}) >= 40, "puzzles repeat")
    widths, prefixes, backgrounds, foregrounds, layouts, ordered_roles = set(), set(), set(), set(), set(), set()
    familiar_width = smaller_width = larger_width = novel_prefix = novel_foreground = full_edge = zero_foreground = 0
    combined = 0
    source_like = wide_before_large = reused_foreground = 0
    nonzero_background = one_cell_test = interior_test = familiar_prefix = familiar_foreground = 0
    change_levels: set[int] = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        tests = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        train_widths = {item[0] for item in train}
        train_prefixes = {item[3] for item in train}
        train_foregrounds = {item[2] for item in train}
        train_max = max(train_widths)
        source_like += train_max == 10 and any(item[0] > 10 for item in tests)
        wide_before_large += train_max >= 18 and any(item[0] >= 28 and item[0] > train_max for item in tests)
        reused_foreground += any(a[2] == b[2] and (a[0], a[3]) != (b[0], b[3]) for i, a in enumerate(train) for b in train[i + 1:])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                width, background, foreground, prefix = _analyze_input(pair["input"])
                widths.add(width)
                prefixes.add(prefix)
                backgrounds.add(background)
                foregrounds.add(foreground)
                layouts.add((width, prefix))
                ordered_roles.add((background, foreground))
                nonzero_background += background != 0
                zero_foreground += foreground == 0
                _require(np.array_equal(pair["output"][0], pair["input"][0]), "row zero differs")
                for row in range(width // 2):
                    expected = [foreground] * (prefix + row) + [background] * (width - prefix - row)
                    _require(pair["output"][row].tolist() == expected, "independent row check failed")
                if split == "test":
                    one_cell_test += prefix == 1
                    interior_test += 1 < prefix < width // 2 + 1
                    familiar_prefix += prefix in train_prefixes
                    familiar_foreground += foreground in train_foregrounds
                    familiar_width += width in train_widths
                    smaller_width += width < train_max
                    larger_width += width > train_max
                    novel_prefix += prefix not in train_prefixes
                    novel_foreground += foreground not in train_foregrounds
                    full_edge += prefix == width // 2 + 1
                    combined += (width not in train_widths and prefix not in train_prefixes and foreground not in train_foregrounds)
                    change_levels.add(sum((width not in train_widths, prefix not in train_prefixes, foreground not in train_foregrounds)))
    _require(len(widths) >= 11 and len(prefixes) >= 12 and len(layouts) >= 70, "structural diversity too narrow")
    _require(len(backgrounds) >= 8 and len(foregrounds) >= 8 and len(ordered_roles) >= 50, "palette diversity too narrow")
    _require(min(familiar_width, smaller_width, larger_width, novel_prefix, novel_foreground, full_edge, zero_foreground, combined) >= 3, "review regimes lack coverage")
    _require(source_like >= 3 and wide_before_large >= 3 and reused_foreground >= 30, "training regimes lack coverage")
    _require(nonzero_background >= 100 and one_cell_test >= 5 and interior_test >= 20, "color or prefix range lacks coverage")
    _require(familiar_prefix >= 10 and familiar_foreground >= 10 and change_levels == {0, 1, 2, 3}, "simple and combined shifts lack coverage")
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(10_000 + index, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "explicit counts ignored")
    for index, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11_000 + index, nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "partial train count ignored")
        _require(ns is None or len(puzzle["test"]) == ns, "partial test count ignored")
    for seed, nt, ns in ((31415, None, None), (271828, 7, 3)):
        _require(_fingerprint(generate_puzzle(seed, nt, ns)) == _fingerprint(generate_puzzle(seed, nt, ns)), "nondeterminism")
    _require(len({_fingerprint(generate_puzzle(50_000 + i, 6, 2)) for i in range(12)}) >= 11, "fixed-count seeds repeat")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (10, 10)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
