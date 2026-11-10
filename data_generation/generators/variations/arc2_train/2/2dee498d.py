"""Generate strips compressed by stable, whole-vector deduplication."""

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_ORIGINAL_PAIR_ROWS = (
    (("200120012001", "421442144214", "412441244124", "100210021002"),
     ("2001", "4214", "4124", "1002")),
    (("451154451", "555555555", "154451154"), ("451", "555", "154")),
    (("212121", "232323"), ("21", "23")),
    (("020440204402044", "220442204422044", "022200222002220",
      "110221102211022", "110201102011020"),
     ("02044", "22044", "02220", "11022", "11020")),
)
_ORIGINAL_PAIRS = tuple(
    (_rows_to_array(source), _rows_to_array(target))
    for source, target in _ORIGINAL_PAIR_ROWS
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _optional_count(num_train, "num_train")
    test = _optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        return minimum + (seed // 4) % (11 - minimum), test
    if train is None:
        minimum = max(2, test + 1)
        return minimum + seed % (11 - minimum), test
    return train, 1 + seed % min(4, train - 1)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(cell) for cell in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _vector_keys(grid: np.ndarray, axis: str) -> list[tuple[int, ...]]:
    if axis == "H":
        return [tuple(int(v) for v in grid[:, j]) for j in range(grid.shape[1])]
    return [tuple(int(v) for v in grid[i, :]) for i in range(grid.shape[0])]


def _compress(grid: np.ndarray, axis: str) -> np.ndarray:
    keys = _vector_keys(grid, axis)
    retained = list(dict.fromkeys(keys))
    if axis == "H":
        return np.array(retained, dtype=np.int8).T.copy()
    return np.array(retained, dtype=np.int8)


def _axis_from_shape(grid: np.ndarray) -> str:
    height, width = grid.shape
    if width >= 2 * height:
        return "H"
    if height >= 2 * width:
        return "V"
    raise AssertionError("input does not have a clear compression axis")


def _is_original_pair(pair: dict) -> bool:
    return any(
        np.array_equal(pair["input"], source)
        and np.array_equal(pair["output"], target)
        for source, target in _ORIGINAL_PAIRS
    )


def _choose_dimensions(rng: np.random.Generator, unique_count: int,
                       repeats: list[int]) -> int:
    total = sum(repeats)
    possible = [v for v in range(2, 11) if total >= 2 * v]
    if int(rng.integers(0, 100)) < 56 and unique_count in possible:
        return unique_count
    return int(rng.choice(possible))


def _make_canonical(rng: np.random.Generator, vector_length: int,
                    unique_count: int) -> np.ndarray:
    minimum_colors = 2
    while minimum_colors ** vector_length < unique_count:
        minimum_colors += 1
    maximum_colors = min(8, vector_length * unique_count)
    palette_size = int(rng.integers(minimum_colors, maximum_colors + 1))
    palette = rng.choice(10, size=palette_size, replace=False).astype(np.int8)
    pattern = int(rng.integers(0, 5))
    repeat_row = pattern == 1 and palette_size ** (vector_length - 1) >= unique_count
    columns: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(unique_count):
        for _attempt in range(4096):
            values = rng.choice(palette, size=vector_length, replace=True)
            if repeat_row:
                values[1] = values[0]
            key = tuple(int(value) for value in values)
            if key not in seen:
                seen.add(key)
                columns.append(key)
                break
        else:
            raise RuntimeError("could not sample distinct whole vectors")
    if pattern in (2, 3) and unique_count >= 2:
        source = int(rng.integers(0, unique_count - 1))
        target = source + 1
        for position in rng.permutation(vector_length):
            for color in rng.permutation(palette):
                altered = list(columns[source])
                altered[int(position)] = int(color)
                key = tuple(altered)
                if key != columns[source] and key not in seen - {columns[target]}:
                    seen.remove(columns[target])
                    seen.add(key)
                    columns[target] = key
                    break
            else:
                continue
            break
    return np.array(columns, dtype=np.int8).T.copy()


def _make_indices(rng: np.random.Generator, unique_count: int,
                  repeats: list[int], layout: str) -> list[int]:
    if layout == "blocks":
        result = list(range(unique_count))
        for _ in range(max(repeats) - 1):
            remaining = [i for i in range(unique_count) if result.count(i) < repeats[i]]
            if int(rng.integers(0, 2)):
                remaining.reverse()
            result.extend(remaining)
        return result

    # Insert a copy before the last new vector appears. The crop is then wrong.
    result = list(range(unique_count))
    selected = int(rng.integers(0, unique_count - 1))
    insertion = int(rng.integers(selected + 1, unique_count))
    result.insert(insertion, selected)
    remaining = []
    for i in range(unique_count):
        remaining.extend([i] * (repeats[i] - result.count(i)))
    if layout == "grouped":
        remaining.sort(reverse=bool(int(rng.integers(0, 2))))
    elif layout == "waves":
        remaining = []
        counts = [repeats[i] - result.count(i) for i in range(unique_count)]
        for wave in range(max(counts)):
            order = list(range(unique_count))
            if wave % 2:
                order.reverse()
            for i in order:
                if counts[i] > wave:
                    remaining.append(i)
    else:
        rng.shuffle(remaining)
    result.extend(remaining)
    return result


def _make_pair(rng: np.random.Generator, axis: str, unique_count: int,
               style: str, harder: bool = False) -> dict:
    for _attempt in range(80):
        if style == "simple":
            count = int(rng.integers(2, 5))
            repeats = [count] * unique_count
            layout = "blocks"
        else:
            upper = 6 if harder else 5
            repeats = [int(rng.integers(2, upper)) for _ in range(unique_count)]
            if style == "interleaved" and int(rng.integers(0, 3)) == 0:
                repeats = [int(rng.integers(2, 6))] * unique_count
            layout = str(rng.choice(("shuffle", "grouped", "waves")))
        if sum(repeats) > 30:
            continue
        vector_length = _choose_dimensions(rng, unique_count, repeats)
        canonical = _make_canonical(rng, vector_length, unique_count)
        indices = _make_indices(rng, unique_count, repeats, layout)
        strip = canonical[:, indices]
        input_grid = strip if axis == "H" else strip.T.copy()
        output_grid = _compress(input_grid, axis)
        return {"input": input_grid.astype(np.int8, copy=False),
                "output": output_grid.astype(np.int8, copy=False)}
    # A bounded deterministic fallback is possible for every requested size.
    repeats = [3] * unique_count
    canonical = _make_canonical(rng, unique_count, unique_count)
    indices = _make_indices(rng, unique_count, repeats,
                            "blocks" if style == "simple" else "waves")
    strip = canonical[:, indices]
    input_grid = strip if axis == "H" else strip.T.copy()
    return {"input": input_grid, "output": _compress(input_grid, axis)}


def _new_pair(rng: np.random.Generator, axis: str, size: int,
              style: str, harder: bool, seen: set[tuple]) -> dict:
    for _ in range(120):
        pair = _make_pair(rng, axis, size, style, harder)
        key = _grid_key(pair["input"])
        if key not in seen and not _is_original_pair(pair):
            seen.add(key)
            return pair
    raise RuntimeError("could not create a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # The only puzzle-wide choice is which axes the demonstrations cover.
    axis_regime = int(rng.integers(0, 3))
    available = ("H",) if axis_regime == 0 else (("V",) if axis_regime == 1 else ("H", "V"))
    if len(available) == 2:
        train_axes = ["H", "V"]
    else:
        train_axes = [available[0], available[0]]
    while len(train_axes) < train_count:
        train_axes.append(str(rng.choice(available)))
    rng.shuffle(train_axes)

    # Source labels also equal first-tile crops. These witnesses explicitly
    # extend that underdetermined evidence to stable global deduplication.
    train_styles = []
    witnessed = set()
    for axis in train_axes:
        if axis not in witnessed:
            train_styles.append("interleaved")
            witnessed.add(axis)
        else:
            train_styles.append("simple" if int(rng.integers(0, 3)) == 0
                                else "interleaved")
    if len(available) == 1 and "simple" not in train_styles:
        train_styles[-1] = "simple"

    train_sizes = [int(rng.integers(2, 9)) for _ in range(train_count)]
    if train_count >= 3 and len(set(train_sizes)) == 1:
        train_sizes[-1] = 2 if train_sizes[-1] > 2 else 3
    seen: set[tuple] = set()
    train = [_new_pair(rng, axis, size, style, False, seen)
             for axis, size, style in zip(train_axes, train_sizes, train_styles)]

    shift = int(rng.integers(0, 3))
    test = []
    for _ in range(test_count):
        if shift == 0:
            size = int(rng.integers(max(train_sizes) + 1, 11))
        elif shift == 1:
            size = int(rng.choice(train_sizes))
        else:
            smaller = [size for size in range(2, max(train_sizes))] or [2]
            size = int(rng.choice(smaller))
        axis = str(rng.choice(available))
        style = "interleaved" if int(rng.integers(0, 5)) else "simple"
        test.append(_new_pair(rng, axis, size, style, True, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} is not a NumPy array")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(all(1 <= dimension <= 30 for dimension in grid.shape),
             f"{label} dimension is outside 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} is not integer")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
             f"{label} has an illegal ARC color")


def _validate_pair(pair: dict, label: str) -> str:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             f"{label} needs exactly input and output")
    source, target = pair["input"], pair["output"]
    _validate_grid(source, f"{label}.input")
    _validate_grid(target, f"{label}.output")
    axis = _axis_from_shape(source)
    _require(not np.array_equal(source, target), f"{label} is unchanged")
    if axis == "H":
        _require(target.shape[0] == source.shape[0]
                 and 2 <= target.shape[1] < source.shape[1],
                 f"{label} horizontal dimensions disagree")
    else:
        _require(target.shape[1] == source.shape[1]
                 and 2 <= target.shape[0] < source.shape[0],
                 f"{label} vertical dimensions disagree")
    source_keys = _vector_keys(source, axis)
    target_keys = _vector_keys(target, axis)
    _require(len(set(target_keys)) == len(target_keys),
             f"{label} retains a duplicate vector")
    _require(all(source_keys.count(key) >= 2 for key in target_keys),
             f"{label} lacks repeated exact vectors")
    _require(np.array_equal(_compress(source, axis), target),
             f"{label} violates stable first-occurrence deduplication")
    _require(set(int(v) for v in source.flat) == set(int(v) for v in target.flat),
             f"{label} changes the palette")
    _require(not _is_original_pair(pair), f"{label} copies a source pair")
    return axis


def _crop_fails(pair: dict, axis: str) -> bool:
    source, target = pair["input"], pair["output"]
    count = target.shape[1] if axis == "H" else target.shape[0]
    crop = source[:, :count] if axis == "H" else source[:count, :]
    return not np.array_equal(crop, target)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    seen_inputs: set[tuple] = set()
    train_axes: set[str] = set()
    witness_axes: set[str] = set()
    test_axes: set[str] = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            axis = _validate_pair(pair, label)
            key = _grid_key(pair["input"])
            _require(key not in seen_inputs, f"{label} repeats an input")
            seen_inputs.add(key)
            if split == "train":
                train_axes.add(axis)
                if _crop_fails(pair, axis):
                    witness_axes.add(axis)
            else:
                test_axes.add(axis)
    _require(test_axes <= train_axes, "test uses an untrained axis")
    _require(witness_axes == train_axes,
             "each trained axis needs a crop-defeating witness")
    _require(any(_crop_fails(pair, _axis_from_shape(pair["input"]))
                 for pair in puzzle["train"]),
             "training does not distinguish stable deduplication from a crop")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    _require(_fingerprint(generate_puzzle(17)) == _fingerprint(automatic[17]),
             "automatic generation is not deterministic")
    _require(len({_fingerprint(puzzle) for puzzle in automatic}) == 42,
             "different seeds generated duplicate puzzles")
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts were ignored")
        _require(_fingerprint(puzzle) ==
                 _fingerprint(generate_puzzle(1000 + train * 10 + test, train, test)),
                 "explicit generation is not deterministic")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(900, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "partial train count was ignored")
        _require(test is None or len(puzzle["test"]) == test,
                 "partial test count was ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2.5, 1), (3, True)):
        try:
            generate_puzzle(3, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")

    train_counts, test_counts = set(), set()
    axes, shapes, palette_sizes = set(), set(), set()
    layouts, repeat_counts, output_shapes = set(), set(), set()
    axis_regimes: set[str] = set()
    orientations: set[str] = set()
    within_palette_variation = 0
    within_shape_variation = 0
    near_matches = 0
    legitimate_duplicate_rows = 0
    early_adjacent = 0
    early_separated = 0
    unequal_counts = 0
    rectangles = 0
    combined = 0
    shifts = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes = set()
        local_shapes = set()
        train_axes = {_axis_from_shape(pair["input"]) for pair in puzzle["train"]}
        axis_regimes.add("mixed" if len(train_axes) == 2 else next(iter(train_axes)))
        train_sizes = [
            len(_vector_keys(pair["output"], _axis_from_shape(pair["input"])))
            for pair in puzzle["train"]
        ]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                axis = _axis_from_shape(source)
                axes.add(axis)
                shapes.add(source.shape)
                output_shapes.add(target.shape)
                local_shapes.add(target.shape)
                palette = tuple(int(c) for c in np.unique(target))
                local_palettes.add(palette)
                palette_sizes.add(len(palette))
                keys = _vector_keys(source, axis)
                target_keys = _vector_keys(target, axis)
                counts = tuple(keys.count(key) for key in target_keys)
                repeat_counts.update(counts)
                layouts.add(("crop" if not _crop_fails(pair, axis) else "interleaved",
                             axis, len(set(counts)) > 1))
                ids = [target_keys.index(key) for key in keys]
                last_new = max(ids.index(index) for index in range(len(target_keys)))
                for index in range(1, last_new + 1):
                    if ids[index] in ids[:index]:
                        if ids[index] == ids[index - 1]:
                            early_adjacent += 1
                        else:
                            early_separated += 1
                if ids[:len(target_keys)] == list(range(len(target_keys))):
                    blocks = [
                        ids[index:index + len(target_keys)]
                        for index in range(0, len(ids), len(target_keys))
                    ]
                    forward = list(range(len(target_keys)))
                    reverse = forward[::-1]
                    if all(block in (forward, reverse) for block in blocks):
                        orientations.add("".join(
                            "F" if block == forward else "R" for block in blocks
                        ))
                unequal = len(set(counts)) > 1
                rectangular = target.shape[0] != target.shape[1]
                unequal_counts += unequal
                rectangles += rectangular
                combined += unequal and rectangular and _crop_fails(pair, axis)
                near_matches += any(
                    sum(a != b for a, b in zip(first, second)) == 1
                    for i, first in enumerate(target_keys)
                    for second in target_keys[i + 1:])
                if axis == "H":
                    legitimate_duplicate_rows += len(set(_vector_keys(target, "V"))) < target.shape[0]
                else:
                    legitimate_duplicate_rows += len(set(_vector_keys(target, "H"))) < target.shape[1]
                retained = list(dict.fromkeys(keys))
                prediction = np.array(retained, dtype=np.int8)
                if axis == "H":
                    prediction = prediction.T
                _require(np.array_equal(prediction, target),
                         "independent visible-vector check failed")
                _require(not _is_original_pair(pair), "original pair was copied")
        within_palette_variation += len(local_palettes) > 1
        within_shape_variation += len(local_shapes) > 1
        for pair in puzzle["test"]:
            axis = _axis_from_shape(pair["input"])
            size = len(_vector_keys(pair["output"], axis))
            shifts.add("larger" if size > max(train_sizes) else
                       ("overlap" if size in train_sizes else "smaller"))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic count distribution is too narrow")
    _require(axes == {"H", "V"} and len(shapes) >= 25
             and len(output_shapes) >= 20, "dimension or axis coverage is too narrow")
    _require(within_palette_variation >= 20 and within_shape_variation >= 20,
             "within-puzzle variation is too narrow")
    _require(max(palette_sizes) >= 5 and {2, 3, 4} <= repeat_counts,
             "palette or multiplicity coverage is too narrow")
    _require(len(layouts) >= 6 and near_matches >= 10
             and legitimate_duplicate_rows >= 5,
             "structural examples are too narrow")
    _require(axis_regimes == {"H", "V", "mixed"}
             and any(code.startswith("FFF") for code in orientations)
             and any(code.startswith("FFR") for code in orientations)
             and any(code.startswith("FRR") for code in orientations),
             "axis regimes or trailing reflection positions are too narrow")
    _require(early_adjacent >= 10 and early_separated >= 10
             and unequal_counts >= 20 and rectangles >= 20 and combined >= 10,
             "early duplicates, multiplicities, or rectangles are too narrow")
    _require(10 in {len(_vector_keys(pair["output"], _axis_from_shape(pair["input"])))
                    for puzzle in automatic for split in ("train", "test")
                    for pair in puzzle[split]},
             "maximum square-family size was not reached")
    _require(shifts == {"larger", "smaller", "overlap"},
             "train/test size relations are too narrow")
