import numpy as np


_SOURCE_PAIR_ROWS = {
    (
        (
            "00000000", "03000000", "00000000", "00000000",
            "00030000", "00000000", "01000000", "00000200",
            "00000000", "00000000",
        ),
        (
            "00000200", "33333333", "00000200", "00000200",
            "33333333", "00000200", "11111111", "00000200",
            "00000200", "00000200",
        ),
    ),
    (
        (
            "000000000", "000000000", "002000000", "000000000",
            "000000030", "000000000", "000100000", "000000000",
            "000000000",
        ),
        (
            "002000000", "002000000", "002000000", "002000000",
            "333333333", "002000000", "111111111", "002000000",
            "002000000",
        ),
    ),
    (
        (
            "00000000000", "01000000000", "00000000000",
            "00000000300", "00000000000", "00000000000",
            "00300000000", "00000000000", "00020000000",
            "00000000020",
        ),
        (
            "00020000020", "11111111111", "00020000020",
            "33333333333", "00020000020", "00020000020",
            "33333333333", "00020000020", "00020000020",
            "00020000020",
        ),
    ),
    (
        (
            "00030000000", "00000000020", "00000000000",
            "00000300000", "00000000000", "00002000000",
            "00000000000", "01000000000", "00000000000",
            "00000000100", "00000000000", "00000000000",
        ),
        (
            "33333333333", "00002000020", "00002000020",
            "33333333333", "00002000020", "00002000020",
            "00002000020", "11111111111", "00002000020",
            "11111111111", "00002000020", "00002000020",
        ),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _checked_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(name + " must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(name + " is outside its legal range")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("no legal count completion exists")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.choice(10, size=3, replace=False)]
    return {
        "horizontals": (colors[0], colors[1]),
        "vertical": colors[2],
        "backgrounds": tuple(color for color in range(10) if color not in colors),
        "allow_overwrite": bool(rng.integers(0, 2)),
        "novel_repeat": int(rng.integers(0, 5)),
    }


def _well_spaced(points: list[tuple[int, int]]) -> bool:
    return all(
        abs(a[0] - b[0]) > 1 or abs(a[1] - b[1]) > 1
        for index, a in enumerate(points)
        for b in points[index + 1:]
    )


def _valid_position(
    position: tuple[int, int],
    role: int,
    placed: list[tuple[tuple[int, int], int]],
    horizontals: tuple[int, int],
    vertical: int,
    allow_overwrite: bool,
) -> bool:
    row, col = position
    for (other_row, other_col), other_role in placed:
        if abs(row - other_row) <= 1 and abs(col - other_col) <= 1:
            return False
        if row == other_row:
            if role in horizontals and other_role in horizontals and role != other_role:
                return False
            if not allow_overwrite and ((role == vertical and other_role in horizontals) or
                                        (other_role == vertical and role in horizontals)):
                return False
    return True


def _place_markers(
    rng: np.random.Generator,
    height: int,
    width: int,
    roles: list[int],
    horizontals: tuple[int, int],
    vertical: int,
    allow_overwrite: bool,
    force_overwrite: bool = False,
) -> list[tuple[int, int]]:
    role_pairs = {
        "horizontal_column": [(i, j) for i in range(len(roles)) for j in range(i + 1, len(roles))
                              if roles[i] in horizontals and roles[j] in horizontals],
        "vertical_row": [(i, j) for i in range(len(roles)) for j in range(i + 1, len(roles))
                         if roles[i] == vertical and roles[j] == vertical],
        "same_horizontal_row": [(i, j) for i in range(len(roles)) for j in range(i + 1, len(roles))
                                if roles[i] in horizontals and roles[i] == roles[j]],
        "same_vertical_column": [(i, j) for i in range(len(roles)) for j in range(i + 1, len(roles))
                                 if roles[i] == vertical and roles[j] == vertical],
        "overwrite": [(i, j) for i in range(len(roles)) for j in range(i + 1, len(roles))
                      if (roles[i] == vertical and roles[j] in horizontals) or
                         (roles[j] == vertical and roles[i] in horizontals)],
    }
    modes = ["none", "none"] + [name for name, pairs in role_pairs.items()
                                if pairs and (name != "overwrite" or allow_overwrite)]
    if force_overwrite:
        modes = ["overwrite"]

    all_cells = [(row, col) for row in range(height) for col in range(width)]
    for _ in range(100):
        mode = modes[int(rng.integers(len(modes)))]
        positions: list[tuple[int, int] | None] = [None] * len(roles)
        placed: list[tuple[tuple[int, int], int]] = []
        if mode != "none":
            pairs = role_pairs[mode]
            i, j = pairs[int(rng.integers(len(pairs)))]
            first = (int(rng.integers(height)), int(rng.integers(width)))
            if mode in ("horizontal_column", "same_vertical_column"):
                alternatives = [(r, first[1]) for r in range(height) if abs(r - first[0]) > 1]
            else:
                alternatives = [(first[0], c) for c in range(width) if abs(c - first[1]) > 1]
            if not alternatives:
                continue
            second = alternatives[int(rng.integers(len(alternatives)))]
            if not _valid_position(second, roles[j], [(first, roles[i])],
                                   horizontals, vertical, allow_overwrite):
                continue
            positions[i], positions[j] = first, second
            placed.extend(((first, roles[i]), (second, roles[j])))

        remaining = [i for i, position in enumerate(positions) if position is None]
        rng.shuffle(remaining)
        for index in remaining:
            role = roles[index]
            preferred: list[tuple[int, int]] = []
            for (row, col), previous_role in placed:
                if role in horizontals and previous_role in horizontals:
                    preferred.extend((r, col) for r in range(height))
                    if role == previous_role:
                        preferred.extend((row, c) for c in range(width))
                if role == vertical and previous_role == vertical:
                    preferred.extend((row, c) for c in range(width))
                    preferred.extend((r, col) for r in range(height))
                if allow_overwrite and ((role == vertical and previous_role in horizontals) or
                                        (role in horizontals and previous_role == vertical)):
                    preferred.extend((row, c) for c in range(width))
            rng.shuffle(preferred)
            shuffled_cells = all_cells.copy()
            rng.shuffle(shuffled_cells)
            candidates = (preferred + shuffled_cells) if int(rng.integers(3)) == 0 else shuffled_cells
            point = next((cell for cell in candidates
                          if _valid_position(cell, role, placed, horizontals,
                                             vertical, allow_overwrite)), None)
            if point is None:
                break
            positions[index] = point
            placed.append((point, role))
        else:
            return [point for point in positions if point is not None]

    raise RuntimeError("could not place sparse markers")


def _construct_output(
    input_grid: np.ndarray,
    background: int,
    vertical: int,
    horizontal_colors: tuple[int, int],
) -> np.ndarray:
    output = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    for _, col in np.argwhere(input_grid == vertical):
        output[:, int(col)] = vertical
    for color in horizontal_colors:
        for row, _ in np.argwhere(input_grid == color):
            output[int(row), :] = color
    return output


def _sample_shape(rng: np.random.Generator, anchor: bool) -> tuple[int, int]:
    if anchor:
        return int(rng.integers(8, 14)), int(rng.integers(8, 15))
    bands = ((6, 11), (11, 19), (19, 31))
    row_band = bands[int(rng.choice(3, p=(0.30, 0.40, 0.30)))]
    col_band = bands[int(rng.choice(3, p=(0.30, 0.40, 0.30)))]
    return int(rng.integers(*row_band)), int(rng.integers(*col_band))


def _sample_role_counts(rng: np.random.Generator, anchor: bool) -> tuple[int, int, int]:
    if anchor:
        return (int(rng.integers(1, 3)), int(rng.integers(1, 3)),
                int(rng.integers(1, 3)))
    for _ in range(100):
        counts = tuple(int(x) for x in rng.integers(0, 4, size=3))
        total = sum(counts)
        if (2 <= total <= 8 and counts[0] + counts[1] >= 1 and
            (counts[2] >= 1 or (counts[0] >= 1 and counts[1] >= 1))):
            return counts
    return (1, 1, 1)


def _generate_pair(
    rng: np.random.Generator,
    parameters: dict,
    background: int,
    anchor: bool,
    section: str,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    horizontals = parameters["horizontals"]
    vertical = parameters["vertical"]
    novelty_index = parameters["novel_repeat"]
    counts = list(_sample_role_counts(rng, anchor))
    if novelty_index < 3 and section == "train":
        counts[novelty_index] = min(counts[novelty_index], 1)
    if novelty_index < 3 and section == "test" and pair_index == 0:
        for _ in range(100):
            counts[novelty_index] = max(counts[novelty_index], 2)
            if sum(counts) <= 8:
                break
            counts = list(_sample_role_counts(rng, False))
        else:
            raise RuntimeError("could not sample test-side repetition")
    roles = [horizontals[0]] * counts[0] + [horizontals[1]] * counts[1] + [vertical] * counts[2]
    rng.shuffle(roles)
    height, width = _sample_shape(rng, anchor)
    positions = _place_markers(rng, height, width, roles, horizontals, vertical,
                               parameters["allow_overwrite"],
                               force_overwrite=anchor and parameters["allow_overwrite"])
    input_grid = np.full((height, width), background, dtype=np.int8)
    for (row, col), role in zip(positions, roles):
        input_grid[row, col] = role
    return input_grid, _construct_output(input_grid, background, vertical, horizontals)


def _value_signature(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.ravel()))


def _row_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return (_row_signature(input_grid), _row_signature(output_grid)) in _SOURCE_PAIR_ROWS


def _background_of(input_grid: np.ndarray) -> int:
    values, counts = np.unique(input_grid, return_counts=True)
    maximum = int(np.max(counts))
    _require(int(np.count_nonzero(counts == maximum)) == 1, "background is ambiguous")
    return int(values[int(np.argmax(counts))])


def _infer_parameters_from_training(train_pairs: list[dict]) -> dict:
    signals = set()
    backgrounds = set()
    for pair in train_pairs:
        background = _background_of(pair["input"])
        backgrounds.add(background)
        signals.update(int(x) for x in np.unique(pair["input"]) if int(x) != background)
    _require(len(signals) == 3, "training must reveal three signal colors")
    _require(not (signals & backgrounds), "background and signal colors overlap")
    valid_verticals = []
    for candidate in sorted(signals):
        horizontals = tuple(sorted(signals - {candidate}))
        if all(np.array_equal(
            _construct_output(pair["input"], _background_of(pair["input"]),
                              candidate, horizontals), pair["output"])
               for pair in train_pairs):
            valid_verticals.append(candidate)
    _require(len(valid_verticals) == 1, "axis roles are not uniquely recoverable")
    return {"horizontals": tuple(sorted(signals - {valid_verticals[0]})),
            "vertical": valid_verticals[0], "backgrounds": backgrounds}


def _validate_marker_geometry(input_grid: np.ndarray, background: int,
                              horizontals: tuple[int, int]) -> None:
    points = [(int(row), int(col)) for row, col in np.argwhere(input_grid != background)]
    _require(_well_spaced(points), "markers are not isolated")
    rows_a = {int(row) for row, _ in np.argwhere(input_grid == horizontals[0])}
    rows_b = {int(row) for row, _ in np.argwhere(input_grid == horizontals[1])}
    _require(not (rows_a & rows_b), "different horizontal colors share a row")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    backgrounds = list(parameters["backgrounds"])
    rng.shuffle(backgrounds)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    previous_background = None
    pair_number = 0
    for section, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            background = backgrounds[pair_number % len(backgrounds)]
            if background == previous_background:
                background = backgrounds[(pair_number + 1) % len(backgrounds)]
            previous_background = background
            pair_number += 1
            for _ in range(100):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                input_grid, output_grid = _generate_pair(
                    child_rng, parameters, background, section == "train" and pair_index == 0,
                    section, pair_index)
                signature = _value_signature(input_grid)
                if signature in seen_inputs or _is_source_pair(input_grid, output_grid):
                    continue
                seen_inputs.add(signature)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a 2D array")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions are illegal")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype is not integer")
                _require(int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9,
                         "grid color is illegal")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "output dimensions or content are invalid")
            signature = _value_signature(inp)
            pair_signature = (signature, _value_signature(out))
            _require(signature not in seen_inputs and pair_signature not in seen_pairs,
                     "duplicate generated pair")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            _require(not _is_source_pair(inp, out), "source pair was copied")

    parameters = _infer_parameters_from_training(puzzle["train"])
    horizontals, vertical = parameters["horizontals"], parameters["vertical"]
    signals = set(horizontals) | {vertical}
    train_backgrounds = parameters["backgrounds"]
    _require(len(train_backgrounds) >= 2, "training backgrounds do not vary")
    train_crossings = set()
    train_overwrite = False
    shapes = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            inp, out = pair["input"], pair["output"]
            background = _background_of(inp)
            _require(background not in signals, "background conflicts with a signal")
            present = set(int(x) for x in np.unique(inp)) - {background}
            _require(present <= signals and present, "unknown or absent signal palette")
            _require(set(int(x) for x in np.unique(out)) <= signals | {background},
                     "output palette is inconsistent")
            _validate_marker_geometry(inp, background, horizontals)
            _require(np.array_equal(out, _construct_output(inp, background, vertical, horizontals)),
                     "incorrect line expansion or crossing priority")
            rows_by_color = {color: {int(row) for row, _ in np.argwhere(inp == color)}
                             for color in horizontals}
            vertical_rows = {int(row) for row, _ in np.argwhere(inp == vertical)}
            vertical_columns = {int(col) for _, col in np.argwhere(inp == vertical)}
            if section == "train":
                for color in horizontals:
                    if rows_by_color[color] and vertical_columns:
                        train_crossings.add(color)
                train_overwrite |= bool(vertical_rows & set.union(*rows_by_color.values()))
            else:
                _require(not (vertical_rows & set.union(*rows_by_color.values())) or train_overwrite,
                         "test overwrites a marker without training demonstration")
            shapes.add(inp.shape)
            _require(2 <= len(np.argwhere(inp != background)) <= 8,
                     "marker count is outside supported sparse regime")
    _require(train_crossings == set(horizontals), "training lacks crossing evidence")
    _require(len(shapes) >= 2, "puzzle has no shape variation")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and all(
        np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
        for a, b in zip(first[section], second[section]) for name in ("input", "output"))
        for section in ("train", "test"))


def _pair_features(pair: dict, horizontals: tuple[int, int], vertical: int) -> dict:
    inp = pair["input"]
    bg = _background_of(inp)
    h, w = inp.shape
    hpoints = [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere(np.isin(inp, horizontals))]
    vpoints = [(int(r), int(c)) for r, c in np.argwhere(inp == vertical)]
    counts = tuple(int(np.count_nonzero(inp == color)) for color in (*horizontals, vertical))
    return {
        "shape": (h, w), "background": bg, "counts": counts,
        "horizontal_column": len({c for _, c, _ in hpoints}) < len(hpoints),
        "vertical_row": len({r for r, _ in vpoints}) < len(vpoints),
        "same_horizontal_row": any(sum(r == other_r and color == other_color
                                       for other_r, _, other_color in hpoints) > 1
                                   for r, _, color in hpoints),
        "same_vertical_column": len({c for _, c in vpoints}) < len(vpoints),
        "overwrite": any(r == hr for r, _ in vpoints for hr, _, _ in hpoints),
        "top_horizontal": any(r == 0 for r, _, _ in hpoints),
        "bottom_horizontal": any(r == h - 1 for r, _, _ in hpoints),
        "left_vertical": any(c == 0 for _, c in vpoints),
        "right_vertical": any(c == w - 1 for _, c in vpoints),
        "marker_count": len(hpoints) + len(vpoints),
        "omitted_role": any(count == 0 for count in counts),
    }


def self_test() -> None:
    train_counts, test_counts, shapes, roles, marker_counts = set(), set(), set(), set(), set()
    features = {section: [] for section in ("train", "test")}
    size_relations = set()
    distinct_puzzles = set()
    first_repetitions_in_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        parameters = _infer_parameters_from_training(puzzle["train"])
        horizontal, vertical = parameters["horizontals"], parameters["vertical"]
        for color in (*horizontal, vertical):
            if (all(np.count_nonzero(pair["input"] == color) <= 1
                    for pair in puzzle["train"]) and
                any(np.count_nonzero(pair["input"] == color) >= 2
                    for pair in puzzle["test"])):
                first_repetitions_in_test += 1
        roles.add((*horizontal, vertical))
        distinct_puzzles.add(tuple((_value_signature(pair["input"]),
                                    _value_signature(pair["output"]))
                                   for section in ("train", "test") for pair in puzzle[section]))
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                f = _pair_features(pair, horizontal, vertical)
                features[section].append(f)
                shapes.add(f["shape"])
                marker_counts.add(f["marker_count"])
                inp, out = pair["input"], pair["output"]
                bg = f["background"]
                for row in range(inp.shape[0]):
                    for col in range(inp.shape[1]):
                        expected = bg
                        if np.any(inp[:, col] == vertical):
                            expected = vertical
                        for color in horizontal:
                            if np.any(inp[row, :] == color):
                                expected = color
                        _require(int(out[row, col]) == expected,
                                 "independent cellwise output check failed")
                if section == "test":
                    area = inp.size
                    size_relations.add("larger" if area > max(train_areas) else
                                       "smaller" if area < min(train_areas) else "comparable")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(shapes) >= 60 and len(roles) >= 30 and len(marker_counts) >= 6,
             "grid, role, or count diversity is too narrow")
    _require(len(distinct_puzzles) >= 40, "different seeds produce repeated puzzles")
    _require(first_repetitions_in_test >= 8,
             "repetition first appearing in test is too rare")
    _require(size_relations == {"larger", "smaller", "comparable"},
             "train/test size relationships are too narrow")
    for name in ("horizontal_column", "vertical_row", "same_horizontal_row",
                 "same_vertical_column", "overwrite", "omitted_role"):
        _require(any(f[name] for section in features for f in features[section]),
                 name + " variation is absent")
    _require(any(f["overwrite"] for f in features["train"]) and
             any(f["overwrite"] for f in features["test"]),
             "overwritten seeds are missing from a split")
    _require(any(f["counts"][0] >= 2 and f["counts"][1] >= 2
                 for f in features["train"]), "both horizontal roles never repeat in training")
    _require(any(f["counts"][2] == 0 for f in features["train"]) and
             any(f["counts"][2] == 0 for f in features["test"]),
             "vertical-role omission is absent from a split")
    _require(any(f["counts"][0] != f["counts"][1] for f in features["test"]) and
             any(f["marker_count"] < 6 for f in features["test"]),
             "test multiplicities lack variation")
    _require(any(f["top_horizontal"] for f in features["test"]) and
             any(not f["top_horizontal"] for f in features["test"]),
             "test top edge is fixed")
    _require(any(min(f["shape"]) <= 8 for section in features for f in features[section]) and
             any(max(f["shape"]) >= 25 for section in features for f in features[section]),
             "canvas sizes lack range")

    _require(_puzzles_equal(generate_puzzle(123456, 7, 3),
                            generate_puzzle(123456, 7, 3)), "explicit generation changed")
    _require(_puzzles_equal(generate_puzzle(654321), generate_puzzle(654321)),
             "automatic generation changed")
    _require(not _puzzles_equal(generate_puzzle(9001, 6, 2),
                                generate_puzzle(9002, 6, 2)), "different seeds agree")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(10000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts ignored")
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "partial test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(12000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts did not raise ValueError")
