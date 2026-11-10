import numpy as np


_TRAIN_COUNT_OPTIONS = (
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)

_TEST_LOW_COUNT_OPTIONS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


# Exact encodings of all source-task pairs. Generated pairs are checked
# against these encodings and rejected if they match.
_SOURCE_PAIR_ENCODINGS = frozenset(
    {
        (
            (
                "0000000000",
                "0330330330",
                "0000000000",
                "0110330330",
                "0000000000",
                "0330880330",
                "0000000000",
                "0330880330",
                "0000000000",
                "0330220220",
                "0000000000",
                "0220330330",
                "0000000000",
            ),
            (3, 1),
            (2, 8, 1),
        ),
        (
            (
                "0000000000000000",
                "0660880880880880",
                "0000000000000000",
                "0880880220660880",
                "0000000000000000",
                "0110880110880880",
                "0000000000000000",
                "0880110880880880",
                "0000000000000000",
                "0880880660880660",
                "0000000000000000",
                "0880880880880880",
                "0000000000000000",
            ),
            (3, 1),
            (6, 1, 2),
        ),
        (
            (
                "0000000000000",
                "0110110880110",
                "0000000000000",
                "0110220110220",
                "0000000000000",
                "0220110110110",
                "0000000000000",
                "0110110880110",
                "0000000000000",
                "0110880110440",
                "0000000000000",
                "0110880110110",
                "0000000000000",
            ),
            (3, 1),
            (8, 2, 4),
        ),
        (
            (
                "0000000000000",
                "0330110110110",
                "0000000000000",
                "0110110440440",
                "0000000000000",
                "0110440110110",
                "0000000000000",
                "0220110110110",
                "0000000000000",
                "0110220110110",
                "0000000000000",
                "0110110110110",
                "0000000000000",
            ),
            (3, 1),
            (4, 2, 3),
        ),
        (
            (
                "0000000000000",
                "0220440220220",
                "0000000000000",
                "0110220440220",
                "0000000000000",
                "0880220220880",
                "0000000000000",
                "0220220110220",
                "0000000000000",
                "0440220220220",
                "0000000000000",
                "0220110220440",
                "0000000000000",
                "0220220440220",
                "0000000000000",
            ),
            (3, 1),
            (4, 1, 8),
        ),
    }
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_optional_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return value


def _resolve_counts(seed, num_train, num_test, rng):
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        num_test = 1 + (seed % 4)
        minimum_train = max(2, num_test + 1)
        number_of_choices = 11 - minimum_train
        num_train = minimum_train + (
            (seed // 4) % number_of_choices
        )
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng):
    """Choose presentation tendencies, not a hidden answer convention."""
    background = int(rng.integers(0, 10))
    palette = tuple(int(x) for x in rng.choice(
        [x for x in range(10) if x != background], 4, replace=False
    ))
    return {
        "background": background,
        "palette": palette,
        "shared_palette": bool(rng.integers(0, 4) == 0),
        "shared_background": bool(rng.integers(0, 4) != 0),
        "novel_test_filler": bool(rng.integers(0, 3) == 0),
        "paired_demonstration": bool(rng.integers(0, 2)),
    }


def _first_accent_order(values, filler):
    order = []
    for value in values:
        color = int(value)
        if color != filler and color not in order:
            order.append(color)
    return tuple(order)


def _mutate_palette(rng, palette, background, required=()):
    required = set(required)
    replaceable = [x for x in palette if x not in required]
    count = min(len(replaceable), int(rng.integers(1, 3)))
    old = set(int(x) for x in rng.choice(replaceable, count, replace=False))
    new = [x for x in range(10) if x not in palette and x != background]
    replacements = [int(x) for x in rng.choice(new, count, replace=False)]
    result = [x for x in palette if x not in old] + replacements
    rng.shuffle(result)
    return tuple(result)


_SHAPES = tuple(
    (rows, columns)
    for rows in range(2, 15)
    for columns in range(2, 10)
    if rows * columns >= 10
)


def _choose_shape(rng, candidates):
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_counts(rng, capacity, regime):
    profiles = [
        (top, middle, low)
        for top in range(3, min(24, capacity) + 1)
        for middle in range(2, top)
        for low in range(1, middle)
        if 2 * top + middle + low + 1 <= capacity
    ]
    mode = int(rng.integers(0, 4))
    if mode == 0:
        subset = [p for p in profiles if p[0] - p[1] == p[1] - p[2] == 1]
    elif mode == 1:
        subset = [p for p in profiles if p[0] - p[1] >= 2 and p[1] - p[2] >= 2]
    elif mode == 2:
        subset = [p for p in profiles if capacity - sum(p) - p[0] <= 2]
    else:
        subset = profiles
    if not subset:
        subset = profiles
    if regime == "target_test" and rng.integers(0, 3) == 0:
        threshold = max(p[0] for p in subset) - 2
        high = [p for p in subset if p[0] >= threshold]
        if high:
            subset = high
    return subset[int(rng.integers(0, len(subset)))]


def _sample_background(rng, palette, parameters, force_change=False):
    base = parameters["background"]
    if parameters["shared_background"]:
        return base
    if force_change or rng.integers(0, 3) == 0:
        choices = [x for x in range(10) if x not in palette and x != base]
        return int(rng.choice(choices))
    return base


def _make_pair(rng, shape, palette, background, filler, counts, clustered):
    rows, columns = shape
    accents = [x for x in palette if x != filler]
    assigned = tuple(int(x) for x in rng.permutation(counts))
    color_counts = dict(zip(accents, assigned))
    ranked = tuple(sorted(accents, key=lambda x: -color_counts[x]))
    filler_count = rows * columns - sum(assigned)
    if filler_count <= max(assigned):
        raise RuntimeError("invalid frequency profile for the selected shape")
    colors = []
    for color, count in color_counts.items():
        colors.extend([color] * count)
    colors.extend([filler] * filler_count)
    values = np.asarray(colors, dtype=np.int8)
    for _ in range(64):
        if clustered:
            order = rng.permutation(len(palette))
            blocks = [[color] * color_counts.get(color, filler_count)
                      for color in (palette[int(i)] for i in order)]
            values = np.asarray([x for block in blocks for x in block], dtype=np.int8)
        else:
            rng.shuffle(values)
        if _first_accent_order(values, filler) != ranked:
            break
    else:
        # A deterministic fallback exists for every distinct triple.
        ordered_accents = tuple(reversed(ranked))
        remaining = dict(color_counts)
        for color in ordered_accents:
            remaining[color] -= 1
        values = np.asarray(
            list(ordered_accents)
            + [color for color in accents for _ in range(remaining[color])]
            + [filler] * filler_count,
            dtype=np.int8,
        )
    if len(values) != rows * columns:
        raise RuntimeError("token count was lost during layout construction")
    tokens = values.reshape(rows, columns)
    grid = np.full((2 * rows + 1, 3 * columns + 1), background, dtype=np.int8)
    for row in range(rows):
        for column in range(columns):
            grid[2 * row + 1, 3 * column + 1:3 * column + 3] = tokens[row, column]
    return grid, np.asarray(ranked, dtype=np.int8).reshape(3, 1)


def _plan_pairs(rng, num_train, num_test, parameters):
    base = parameters["palette"]
    background = parameters["background"]
    first_filler = int(rng.choice(base))
    reserved = int(rng.choice([x for x in base if x != first_filler]))
    initial_shape = _choose_shape(rng, _SHAPES)
    plans = []
    train_shapes = []
    first_counts = None
    for index in range(num_train):
        if index == 0 or parameters["shared_palette"] or index == 2:
            palette = base
        else:
            palette = _mutate_palette(rng, base, background, required=(first_filler,))
        if index == 0 or index == 2:
            filler = first_filler
        else:
            candidates = [x for x in palette if x != first_filler and
                          (not parameters["novel_test_filler"] or x != reserved)]
            filler = int(rng.choice(candidates))
        if index == 0:
            shape = initial_shape
        elif index == 1 and num_train >= 3:
            shape = _choose_shape(rng, [s for s in _SHAPES if s[1] == initial_shape[1]
                                        and s[0] != initial_shape[0]])
        elif index == 2:
            alternatives = [s for s in _SHAPES if s[1] != initial_shape[1]]
            if not parameters["paired_demonstration"]:
                same_capacity = [s for s in alternatives
                                 if s[0] * s[1] == initial_shape[0] * initial_shape[1]]
                if same_capacity:
                    alternatives = same_capacity
            shape = _choose_shape(rng, alternatives)
        elif index == 1 and num_train == 2:
            shape = _choose_shape(rng, [s for s in _SHAPES if s != initial_shape])
        else:
            shape = _choose_shape(rng, _SHAPES)
        capacity = shape[0] * shape[1]
        if index == 2 and first_counts is not None and sum(first_counts) + max(first_counts) + 1 <= capacity:
            counts = first_counts
        else:
            counts = _sample_counts(rng, capacity, "source_train")
        if index == 0:
            first_counts = counts
        plans.append(("train", shape, palette,
                      _sample_background(rng, palette, parameters, force_change=index == 1),
                      filler, counts, index == 2 and not parameters["paired_demonstration"]))
        train_shapes.append(shape)
    for index in range(num_test):
        anchor = train_shapes[int(rng.integers(0, len(train_shapes)))]
        geometry_mode = int(rng.integers(0, 4))
        if geometry_mode == 0:
            candidates = [anchor]
        elif geometry_mode == 1:
            candidates = [s for s in _SHAPES if s[0] <= anchor[0] and s[1] <= anchor[1]]
        elif geometry_mode == 2:
            candidates = [s for s in _SHAPES if s[0] >= anchor[0] and s[1] >= anchor[1]]
        else:
            candidates = _SHAPES
        shape = _choose_shape(rng, candidates)
        if index == 0 and parameters["novel_test_filler"]:
            palette = base
            filler = reserved
        else:
            if parameters["shared_palette"] or rng.integers(0, 3) == 0:
                palette = base
            else:
                palette = _mutate_palette(rng, base, background, required=(first_filler,))
            if index > 0 and rng.integers(0, 4) != 0:
                prior_test_fillers = {p[4] for p in plans if p[0] == "test"}
                choices = [x for x in palette if x not in prior_test_fillers]
                filler = int(rng.choice(choices or list(palette)))
            else:
                filler = first_filler if first_filler in palette else int(rng.choice(palette))
        counts = _sample_counts(rng, shape[0] * shape[1], "target_test")
        plans.append(("test", shape, palette,
                      _sample_background(rng, palette, parameters),
                      filler, counts, bool(rng.integers(0, 4) == 0)))
    return plans

def _array_signature(array):
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _pair_encoding(input_grid, output_grid):
    input_rows = tuple(
        "".join(str(int(value)) for value in row)
        for row in input_grid
    )
    return (
        input_rows,
        tuple(int(x) for x in output_grid.shape),
        tuple(int(x) for x in output_grid.ravel()),
    )


def _is_source_pair(input_grid, output_grid):
    return (
        _pair_encoding(input_grid, output_grid)
        in _SOURCE_PAIR_ENCODINGS
    )


def _analyze_input(input_grid):
    height, width = input_grid.shape

    _require(
        (height - 1) % 2 == 0,
        "input height is incompatible with separated token rows",
    )
    _require(
        (width - 1) % 3 == 0,
        "input width is incompatible with two-cell tokens",
    )

    token_rows = (height - 1) // 2
    token_columns = (width - 1) // 3

    _require(token_rows >= 1, "input has no token rows")
    _require(token_columns >= 1, "input has no token columns")

    background = int(input_grid[0, 0])

    _require(
        np.all(input_grid[0::2, :] == background),
        "separator or border row is malformed",
    )
    _require(
        np.all(input_grid[:, 0::3] == background),
        "separator or border column is malformed",
    )

    tokens = []
    token_counts = {}

    for row in range(token_rows):
        grid_row = 2 * row + 1
        for column in range(token_columns):
            grid_column = 3 * column + 1
            left = int(input_grid[grid_row, grid_column])
            right = int(input_grid[grid_row, grid_column + 1])

            _require(
                left == right,
                "a token is not a monochromatic two-cell block",
            )
            _require(
                left != background,
                "background color appears inside a token",
            )

            tokens.append(left)
            token_counts[left] = token_counts.get(left, 0) + 1

    _require(
        len(token_counts) == 4,
        "input must contain exactly four token colors",
    )

    expected_colors = {background} | set(token_counts)
    actual_colors = {
        int(color) for color in np.unique(input_grid)
    }
    _require(
        actual_colors == expected_colors,
        "input contains an unexpected color",
    )

    greatest_count = max(token_counts.values())
    fillers = [
        color
        for color, count in token_counts.items()
        if count == greatest_count
    ]
    _require(
        len(fillers) == 1,
        "filler color must be uniquely most frequent",
    )

    filler = fillers[0]
    accent_colors = tuple(
        color for color in token_counts if color != filler
    )
    accent_values = [
        token_counts[color] for color in accent_colors
    ]

    _require(
        len(set(accent_values)) == 3,
        "accent frequencies must be pairwise distinct",
    )
    _require(
        greatest_count > max(accent_values),
        "filler must be strictly more frequent than every accent",
    )

    ranked_accents = tuple(
        sorted(
            accent_colors,
            key=lambda color: token_counts[color],
            reverse=True,
        )
    )

    return {
        "background": background,
        "token_rows": token_rows,
        "token_columns": token_columns,
        "tokens": tuple(tokens),
        "token_counts": dict(token_counts),
        "token_colors": frozenset(token_counts),
        "filler": filler,
        "accents": frozenset(accent_colors),
        "ranked_accents": ranked_accents,
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test, rng)
    parameters = sample_puzzle_parameters(rng)
    plans = _plan_pairs(rng, num_train, num_test, parameters)
    stream_entropy = int(rng.integers(0, np.iinfo(np.int64).max))
    child_sequences = np.random.SeedSequence([seed, stream_entropy]).spawn(len(plans))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_counts = plans[0][5]
    for index, (plan, child) in enumerate(zip(plans, child_sequences)):
        pair_rng = np.random.default_rng(child)
        split, shape, palette, background, filler, counts, clustered = plan
        for _ in range(128):
            input_grid, output_grid = _make_pair(
                pair_rng, shape, palette, background, filler, counts, clustered
            )
            if index == 0:
                output_colors = tuple(int(x) for x in output_grid.ravel())
                if output_colors in (tuple(sorted(output_colors)),
                                     tuple(sorted(output_colors, reverse=True))):
                    continue
                first_order = _first_accent_order(
                    input_grid[1::2, 1::3].ravel(), filler
                )
                if output_colors == tuple(reversed(first_order)):
                    continue
            if _array_signature(input_grid) in seen_inputs or _is_source_pair(input_grid, output_grid):
                continue
            if index == 2 and counts == first_counts and palette == plans[0][2] and filler == plans[0][4]:
                same_ranking = np.array_equal(output_grid, puzzle["train"][0]["output"])
                if same_ranking != clustered:
                    continue
            seen_inputs.add(_array_signature(input_grid))
            puzzle[split].append({"input": input_grid, "output": output_grid})
            break
        else:
            raise RuntimeError("could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "malformed pair")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid contains illegal ARC colors")
            inp, out = pair["input"], pair["output"]
            input_sig, output_sig = _array_signature(inp), _array_signature(out)
            _require(input_sig not in seen_inputs, "duplicate input")
            _require((input_sig, output_sig) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_sig)
            seen_pairs.add((input_sig, output_sig))
            _require(not _is_source_pair(inp, out), "source pair was copied")
            _require(out.shape == (3, 1), "output must be a vertical three-cell column")
            info = _analyze_input(inp)
            _require(info["token_rows"] >= 2 and info["token_columns"] >= 2,
                     "token lattice is too small")
            _require(info["token_rows"] * info["token_columns"] >= 10,
                     "too few token slots for distinct counts")
            _require(tuple(int(x) for x in out.ravel()) == info["ranked_accents"],
                     "output does not rank the accent token counts")
            _require(_first_accent_order(info["tokens"], info["filler"])
                     != info["ranked_accents"],
                     "first-occurrence shortcut fits a pair")
            infos[split].append(info)
    train = infos["train"]
    _require(len({i["filler"] for i in train}) >= 2,
             "training does not vary the filler role")
    _require(len({(i["token_rows"], i["token_columns"]) for i in train}) >= 2,
             "training does not vary geometry")
    _require(any(i["ranked_accents"] not in
                 (tuple(sorted(i["ranked_accents"])),
                  tuple(sorted(i["ranked_accents"], reverse=True)))
                 for i in train),
             "training permits a numeric color-order shortcut")
    _require(any(i["ranked_accents"] !=
                 tuple(reversed(_first_accent_order(i["tokens"], i["filler"])))
                 for i in train),
             "training permits a reverse first-occurrence shortcut")
    if n_train >= 3:
        _require(any(a["token_columns"] == b["token_columns"] and a["filler"] != b["filler"]
                     for a in train for b in train),
                 "training lacks repeated width with different fillers")
        _require(any(a["token_columns"] != b["token_columns"] and a["filler"] == b["filler"]
                     for a in train for b in train),
                 "training lacks different widths with the same filler")
    # There is no hidden color map or seed-wide palette. Every test answer is
    # reconstructed from that test input, using the ranking rule demonstrated
    # by the training outputs.


def _puzzle_signature(puzzle):
    return tuple(
        (split, tuple((_array_signature(p["input"]), _array_signature(p["output"]))
                      for p in puzzle[split]))
        for split in ("train", "test")
    )


def _normalized_layout(info):
    mapping = {info["filler"]: 0}
    mapping.update({color: rank for rank, color in enumerate(info["ranked_accents"], 1)})
    return (info["token_rows"], info["token_columns"],
            tuple(mapping[color] for color in info["tokens"]))


def _visible_frequency_rank(grid):
    counts = {}
    for row in range(1, grid.shape[0], 2):
        for column in range(1, grid.shape[1], 3):
            color = int(grid[row, column])
            counts[color] = counts.get(color, 0) + 1
    ordered = sorted(counts, key=lambda color: counts[color], reverse=True)
    return tuple(ordered[1:])


def _same_token_adjacency(info):
    rows, columns = info["token_rows"], info["token_columns"]
    tokens = info["tokens"]
    equal = sum(tokens[r * columns + c] == tokens[r * columns + c + 1]
                for r in range(rows) for c in range(columns - 1))
    equal += sum(tokens[r * columns + c] == tokens[(r + 1) * columns + c]
                 for r in range(rows - 1) for c in range(columns))
    return equal / (rows * (columns - 1) + (rows - 1) * columns)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42,
             "cross-seed puzzle diversity collapsed")
    _require(_puzzle_signature(generate_puzzle(17)) == _puzzle_signature(puzzles[17]),
             "automatic generation is not deterministic")
    explicit = generate_puzzle(7001, 7, 3)
    _require(_puzzle_signature(explicit) == _puzzle_signature(generate_puzzle(7001, 7, 3)),
             "explicit generation is not deterministic")
    extra = [explicit]
    for seed, nt, ne in ((7101, 2, 1), (7102, 10, 1), (7103, 10, 4),
                         (7201, None, 1), (7202, None, 4),
                         (7203, 2, None), (7204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        _require(nt is None or len(puzzle["train"]) == nt, "train count not honored")
        _require(ne is None or len(puzzle["test"]) == ne, "test count not honored")
        extra.append(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_train": 3.0}):
        try:
            generate_puzzle(7301, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count was accepted")
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "too few automatic train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "too few automatic test counts")
    shapes, layouts, profiles, palette_sets = set(), set(), {"train": set(), "test": set()}, set()
    changed_palettes = changed_backgrounds = multi_test_fillers = shared_palettes = 0
    familiar_test_fillers = novel_test_fillers = 0
    equal_geometry = smaller_tests = taller_tests = wider_tests = 0
    close_races = wide_gaps = tight_fillers = 0
    paired_same = paired_different = cluster_contrast = 0
    source_like_shifts = compact_grids = large_grids = 0
    higher_test_counts = lower_test_counts = familiar_test_counts = 0
    for puzzle in puzzles + extra:
        validate_generated_puzzle(puzzle)
        train_infos = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test_infos = [_analyze_input(p["input"]) for p in puzzle["test"]]
        all_infos = train_infos + test_infos
        palette_sets.update(i["token_colors"] for i in all_infos)
        changed_palettes += len({i["token_colors"] for i in all_infos}) > 1
        shared_palettes += len({i["token_colors"] for i in all_infos}) == 1
        changed_backgrounds += len({i["background"] for i in all_infos}) > 1
        multi_test_fillers += len({i["filler"] for i in test_infos}) > 1
        training_fillers = {i["filler"] for i in train_infos}
        familiar_test_fillers += any(i["filler"] in training_fillers for i in test_infos)
        novel_test_fillers += any(i["filler"] not in training_fillers for i in test_infos)
        source_like_shifts += any(
            i["filler"] not in training_fillers
            and any(i["filler"] in t["accents"] and t["filler"] in i["accents"]
                    for t in train_infos)
            for i in test_infos
        )
        train_top_counts = [max(t["token_counts"][x] for x in t["accents"])
                            for t in train_infos]
        if len(train_infos) >= 3 and train_infos[0]["token_colors"] == train_infos[2]["token_colors"] \
                and train_infos[0]["filler"] == train_infos[2]["filler"]:
            if train_infos[0]["ranked_accents"] == train_infos[2]["ranked_accents"]:
                paired_same += 1
                if (all(train_infos[0]["token_counts"][x] ==
                        train_infos[2]["token_counts"][x]
                        for x in train_infos[0]["token_colors"])
                    and _same_token_adjacency(train_infos[2]) >
                        _same_token_adjacency(train_infos[0])):
                    cluster_contrast += 1
            else:
                paired_different += 1
        for info in test_infos:
            top = max(info["token_counts"][x] for x in info["accents"])
            higher_test_counts += top > max(train_top_counts)
            lower_test_counts += top < min(train_top_counts)
            familiar_test_counts += top in train_top_counts
            equal_geometry += any((info["token_rows"], info["token_columns"]) ==
                                  (t["token_rows"], t["token_columns"]) for t in train_infos)
            smaller_tests += any(info["token_rows"] <= t["token_rows"] and
                                 info["token_columns"] <= t["token_columns"] for t in train_infos)
            taller_tests += any(info["token_rows"] > t["token_rows"] for t in train_infos)
            wider_tests += any(info["token_columns"] > t["token_columns"] for t in train_infos)
        for split, infos in (("train", train_infos), ("test", test_infos)):
            for pair, info in zip(puzzle[split], infos):
                shapes.add((info["token_rows"], info["token_columns"]))
                layouts.add(_normalized_layout(info))
                profile = tuple(sorted((info["token_counts"][x] for x in info["accents"]), reverse=True))
                profiles[split].add(profile)
                close_races += profile[0] - profile[1] == profile[1] - profile[2] == 1
                wide_gaps += profile[0] - profile[1] >= 2 and profile[1] - profile[2] >= 2
                tight_fillers += info["token_counts"][info["filler"]] - profile[0] == 1
                compact_grids += (info["token_rows"], info["token_columns"]) == (2, 5)
                large_grids += info["token_rows"] >= 10 and info["token_columns"] >= 7
                _require(tuple(int(x) for x in pair["output"].ravel())
                         == _visible_frequency_rank(pair["input"]),
                         "independent count reconstruction failed")
    _require(len(shapes) >= 35 and len(layouts) >= 150, "structural diversity is insufficient")
    _require(len(profiles["train"]) >= 20 and len(profiles["test"]) >= 15,
             "frequency diversity is insufficient")
    _require(len(palette_sets) >= 30 and changed_palettes >= 15 and changed_backgrounds >= 5,
             "pair-level color diversity is insufficient")
    _require(shared_palettes >= 5 and source_like_shifts >= 5,
             "shared palettes or source-like role shifts are missing")
    _require(multi_test_fillers >= 4 and familiar_test_fillers >= 15 and novel_test_fillers >= 5,
             "test role shifts are insufficient")
    _require(equal_geometry >= 5 and smaller_tests >= 10 and taller_tests >= 10 and wider_tests >= 10,
             "train/test geometry relationships are insufficient")
    _require(close_races >= 20 and wide_gaps >= 20 and tight_fillers >= 5,
             "frequency regime coverage is insufficient")
    _require(compact_grids >= 1 and large_grids >= 10,
             "compact and large arrays are not both represented")
    _require(higher_test_counts >= 5 and lower_test_counts >= 3
             and familiar_test_counts >= 5,
             "test frequencies do not cover higher, lower, and familiar regimes")
    _require(paired_same >= 3 and paired_different >= 3 and cluster_contrast >= 3,
             "paired training comparisons are insufficient")
