import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "00050005000",
            "03050105040",
            "00050005000",
        ),
        (
            "88856665999",
            "88856665999",
            "88856665999",
        ),
    ),
    (
        (
            "00050005000",
            "02050305010",
            "00050005000",
        ),
        (
            "77758885666",
            "77758885666",
            "77758885666",
        ),
    ),
    (
        (
            "00050005000",
            "04050105020",
            "00050005000",
            "55555555555",
            "00050005000",
            "02050305040",
            "00050005000",
        ),
        (
            "99956665777",
            "99956665777",
            "99956665777",
            "55555555555",
            "77758885999",
            "77758885999",
            "77758885999",
        ),
    ),
    (
        (
            "00050005000",
            "01050205010",
            "00050005000",
        ),
        (
            "66657775666",
            "66657775666",
            "66657775666",
        ),
    ),
    (
        (
            "00050005000",
            "02050305040",
            "00050005000",
            "55555555555",
            "00050005000",
            "01050105030",
            "00050005000",
        ),
        (
            "77758885999",
            "77758885999",
            "77758885999",
            "55555555555",
            "66656665888",
            "66656665888",
            "66656665888",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    return int(seed)


def _checked_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count("num_train", num_train, 2, 10)
    test_count = _checked_count("num_test", num_test, 1, 4)

    if (
        train_count is not None
        and test_count is not None
        and train_count <= test_count
    ):
        raise ValueError("num_train must be greater than num_test")

    if train_count is None and test_count is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        train_count = 2 + (seed % 9)
        maximum_test = min(4, train_count - 1)
        test_count = 1 + ((seed // 9) % maximum_test)
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train exists for num_test")
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("illegal train/test count combination")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(x) for x in rng.permutation(10))
    code_count = int(rng.integers(2, 5))
    markers = colors[:code_count]
    fills = colors[code_count:2 * code_count]
    neutral = colors[2 * code_count:]
    return {
        "markers": markers,
        "mapping": dict(zip(markers, fills)),
        "neutral": neutral,
        "vary_neutral": bool(rng.integers(0, 2)),
    }


def _sample_lengths(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    lengths = []
    for index in range(count):
        remaining = count - index - 1
        limit = 30 - (count - 1) - sum(lengths) - 3 * remaining
        options = [x for x in (3, 5, 7, 9, 11) if x <= limit]
        weights = np.array([3.0 / (x - 1) for x in options])
        weights /= weights.sum()
        lengths.append(int(rng.choice(options, p=weights)))
    return tuple(lengths)


def _sample_geometry(
    rng: np.random.Generator, minimum_cells: int = 2, harder: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    options = [
        (rows, cols)
        for rows in range(1, 6)
        for cols in range(1, 6)
        if rows * cols >= minimum_cells
    ]
    count_weights = (0.0, 1.4, 1.8, 1.7, 1.0, 0.65)
    weights = np.array([
        count_weights[rows] * count_weights[cols]
        * (1.5 if harder and (rows >= 3 or cols >= 4) else 1.0)
        for rows, cols in options
    ], dtype=float)
    weights /= weights.sum()
    rows, cols = options[int(rng.choice(len(options), p=weights))]
    return _sample_lengths(rng, rows), _sample_lengths(rng, cols)


def _sample_roles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    code_count: int,
    cover_all: bool = False,
    force_single: bool = False,
) -> tuple[tuple[int, ...], ...]:
    rows, cols = shape
    size = rows * cols
    if force_single:
        roles = np.full(size, int(rng.integers(code_count)), dtype=int)
    else:
        concentration = float(rng.choice((0.35, 0.8, 1.5, 3.0)))
        weights = rng.dirichlet(np.full(code_count, concentration))
        roles = rng.choice(code_count, size=size, p=weights)
    if cover_all:
        positions = rng.choice(size, size=code_count, replace=False)
        for role, position in enumerate(positions):
            roles[int(position)] = role
    return tuple(
        tuple(int(x) for x in roles[row * cols:(row + 1) * cols])
        for row in range(rows)
    )


def _choose_neutral(
    rng: np.random.Generator, parameters: dict, pair_index: int
) -> tuple[int, int]:
    pool = parameters["neutral"]
    if not parameters["vary_neutral"] or pair_index == 0:
        return pool[0], pool[1]
    if pair_index == 1:
        return pool[1], pool[0]
    chosen = rng.choice(pool, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _render_pair(
    geometry: tuple[tuple[int, ...], tuple[int, ...]],
    roles: tuple[tuple[int, ...], ...],
    parameters: dict,
    neutral: tuple[int, int],
) -> dict:
    row_lengths, col_lengths = geometry
    height = sum(row_lengths) + len(row_lengths) - 1
    width = sum(col_lengths) + len(col_lengths) - 1
    background, separator = neutral
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    row_start = 0
    for i, row_length in enumerate(row_lengths):
        col_start = 0
        for j, col_length in enumerate(col_lengths):
            marker = parameters["markers"][roles[i][j]]
            fill = parameters["mapping"][marker]
            inp[row_start + row_length // 2, col_start + col_length // 2] = marker
            out[row_start:row_start + row_length,
                col_start:col_start + col_length] = fill
            col_start += col_length
            if j + 1 < len(col_lengths):
                inp[:, col_start] = separator
                out[:, col_start] = separator
                col_start += 1
        row_start += row_length
        if i + 1 < len(row_lengths):
            inp[row_start, :] = separator
            out[row_start, :] = separator
            row_start += 1
    return {"input": inp, "output": out}


def _grid_signature(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _grid_matches_rows(grid: np.ndarray, rows: tuple[str, ...]) -> bool:
    return grid.shape == (len(rows), len(rows[0])) and all(
        int(grid[i, j]) == int(value)
        for i, row in enumerate(rows)
        for j, value in enumerate(row)
    )


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return any(
        _grid_matches_rows(inp, source_in)
        and _grid_matches_rows(out, source_out)
        for source_in, source_out in _SOURCE_PAIR_ROWS
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value % (2**64))
    train_count, test_count = _resolve_counts(seed_value, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    code_count = len(parameters["markers"])
    fixed_geometry = (
        _sample_geometry(rng, max(9, code_count))
        if rng.random() < 0.18 else None
    )
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_layouts = set()
    train_geometries = []
    total = train_count + test_count
    for pair_index in range(total):
        is_train = pair_index < train_count
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        neutral = _choose_neutral(pair_rng, parameters, pair_index)
        for attempt in range(60):
            if fixed_geometry is not None:
                geometry = fixed_geometry
            elif (
                not is_train and train_geometries
                and pair_rng.random() < 0.35
            ):
                geometry = train_geometries[int(pair_rng.integers(len(train_geometries)))]
            else:
                geometry = _sample_geometry(
                    pair_rng,
                    code_count if pair_index == 0 else 2,
                    harder=not is_train,
                )
            if pair_index == 1 and fixed_geometry is None and attempt < 30:
                if geometry == train_geometries[0]:
                    continue
            shape = (len(geometry[0]), len(geometry[1]))
            roles = _sample_roles(
                pair_rng, shape, code_count,
                cover_all=pair_index == 0,
                force_single=(pair_index > 0 and pair_rng.random() < 0.22),
            )
            layout_key = (geometry, roles)
            if layout_key in seen_layouts:
                continue
            pair = _render_pair(geometry, roles, parameters, neutral)
            key = _grid_signature(pair["input"])
            if key in seen_inputs or _is_source_pair(pair["input"], pair["output"]):
                continue
            seen_layouts.add(layout_key)
            seen_inputs.add(key)
            puzzle["train" if is_train else "test"].append(pair)
            if is_train:
                train_geometries.append(geometry)
            break
        else:
            raise AssertionError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: object, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= x <= 30 for x in grid.shape), f"{label} dimensions invalid")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} dtype invalid")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
             f"{label} colors invalid")


def _runs(length: int, lines: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
    runs = []
    start = 0
    for line in lines:
        if line == 0 or line >= length - 1 or line - start < 3:
            return None
        runs.append((start, line))
        start = line + 1
    if length - start < 3:
        return None
    runs.append((start, length))
    if any((end - start) % 2 == 0 for start, end in runs):
        return None
    return tuple(runs)


def _parse_input(inp: np.ndarray) -> dict:
    height, width = inp.shape
    candidates = []
    for value in (int(x) for x in np.unique(inp)):
        row_lines = tuple(i for i in range(height) if bool(np.all(inp[i, :] == value)))
        col_lines = tuple(j for j in range(width) if bool(np.all(inp[:, j] == value)))
        if not row_lines and not col_lines:
            continue
        row_runs = _runs(height, row_lines)
        col_runs = _runs(width, col_lines)
        if row_runs is None or col_runs is None:
            continue
        background = int(inp[row_runs[0][0], col_runs[0][0]])
        if background == value:
            continue
        markers = []
        valid = True
        for r0, r1 in row_runs:
            row_markers = []
            for c0, c1 in col_runs:
                block = inp[r0:r1, c0:c1]
                marker = int(block[(r1 - r0) // 2, (c1 - c0) // 2])
                expected = np.full(block.shape, background, dtype=inp.dtype)
                expected[(r1 - r0) // 2, (c1 - c0) // 2] = marker
                if marker in (background, value) or not np.array_equal(block, expected):
                    valid = False
                    break
                row_markers.append(marker)
            if not valid:
                break
            markers.append(tuple(row_markers))
        if valid:
            candidates.append({
                "background": background, "separator": value,
                "row_runs": row_runs, "col_runs": col_runs,
                "markers": tuple(markers),
            })
    _require(len(candidates) == 1, "input has ambiguous or invalid dividers")
    return candidates[0]


def _inspect_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input/output dimensions differ")
    info = _parse_input(inp)
    separator = info["separator"]
    row_lines = set(range(inp.shape[0])) - {
        r for start, end in info["row_runs"] for r in range(start, end)
    }
    col_lines = set(range(inp.shape[1])) - {
        c for start, end in info["col_runs"] for c in range(start, end)
    }
    for row in row_lines:
        _require(bool(np.all(out[row, :] == separator)), "output row divider changed")
    for col in col_lines:
        _require(bool(np.all(out[:, col] == separator)), "output column divider changed")
    observations = []
    for i, (r0, r1) in enumerate(info["row_runs"]):
        for j, (c0, c1) in enumerate(info["col_runs"]):
            block = out[r0:r1, c0:c1]
            fill = int(block[0, 0])
            _require(bool(np.all(block == fill)), "output block is not solid")
            _require(fill not in (info["background"], separator, info["markers"][i][j]),
                     "output fill conflicts with an input role")
            observations.append((info["markers"][i][j], fill))
    info["observations"] = tuple(observations)
    info["row_lengths"] = tuple(end - start for start, end in info["row_runs"])
    info["col_lengths"] = tuple(end - start for start, end in info["col_runs"])
    return info


def _training_mapping(train_pairs: list[dict]) -> dict[int, int]:
    mapping = {}
    for pair in train_pairs:
        for marker, fill in _inspect_pair(pair)["observations"]:
            _require(marker not in mapping or mapping[marker] == fill,
                     "training codebook contradicts itself")
            mapping[marker] = fill
    return mapping


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    mapping = _training_mapping(train_pairs)
    info = _parse_input(test_input)
    result = test_input.copy()
    for i, (r0, r1) in enumerate(info["row_runs"]):
        for j, (c0, c1) in enumerate(info["col_runs"]):
            marker = info["markers"][i][j]
            _require(marker in mapping, "test marker unseen in training")
            result[r0:r1, c0:c1] = mapping[marker]
    return result


def _validate_and_describe(puzzle: dict) -> dict:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys invalid")
    _require(all(isinstance(puzzle[x], list) for x in ("train", "test")),
             "puzzle sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "pair counts invalid")
    seen_inputs = set()
    seen_pairs = set()
    seen_layouts = set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{section}[{index}] keys invalid")
            _check_grid(pair["input"], f"{section}[{index}].input")
            _check_grid(pair["output"], f"{section}[{index}].output")
            inp_key = _grid_signature(pair["input"])
            pair_key = (inp_key, _grid_signature(pair["output"]))
            _require(inp_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "input equals output")
            _require(not _is_source_pair(pair["input"], pair["output"]),
                     "source pair copied")
            info = _inspect_pair(pair)
            layout_key = (info["row_lengths"], info["col_lengths"], info["markers"])
            _require(layout_key not in seen_layouts,
                     "repeated compartment layout and marker arrangement")
            seen_layouts.add(layout_key)
            infos[section].append(info)
    mapping = _training_mapping(puzzle["train"])
    _require(2 <= len(mapping) <= 4, "codebook size must be two to four")
    _require(len(set(mapping.values())) == len(mapping), "fills must be distinct")
    role_colors = set(mapping) | set(mapping.values())
    _require(len(role_colors) == 2 * len(mapping), "markers and fills overlap")
    for section in ("train", "test"):
        for info in infos[section]:
            _require(info["background"] != info["separator"],
                     "background and divider coincide")
            _require(not ({info["background"], info["separator"]} & role_colors),
                     "neutral color overlaps codebook")
            for marker, fill in info["observations"]:
                _require(marker in mapping and mapping[marker] == fill,
                         "pair violates the learned codebook")
    for pair in puzzle["test"]:
        prediction = _predict_from_training(puzzle["train"], pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]),
                 "test is not predictable from training")
    return {"mapping": mapping, "train_infos": infos["train"],
            "test_infos": infos["test"]}


def validate_generated_puzzle(puzzle: dict) -> None:
    _validate_and_describe(puzzle)


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(
        len(a[section]) == len(b[section]) and all(
            np.array_equal(x[role], y[role])
            for x, y in zip(a[section], b[section])
            for role in ("input", "output")
        ) for section in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    code_counts, shapes, row_counts, col_counts = set(), set(), set(), set()
    stable_neutral = variable_neutral = 0
    unequal = rectangular = single_column = wide = tall = 0
    train_single_marker = train_adjacent = train_three_distinct = 0
    train_identical_band = test_adjacent = test_nonadjacent = 0
    same_size_test = changed_size_test = 0
    train_many_bands = test_many_bands = new_test_first_band = 0
    combined_variations = 0
    rare_training_code = 0
    fixed_size_puzzles = mixed_size_puzzles = 0
    all_input_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        info = _validate_and_describe(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        code_counts.add(len(info["mapping"]))
        all_infos = info["train_infos"] + info["test_infos"]
        neutral_pairs = {(x["background"], x["separator"]) for x in all_infos}
        stable_neutral += len(neutral_pairs) == 1
        variable_neutral += len(neutral_pairs) > 1
        pair_shapes = {pair["input"].shape for section in ("train", "test")
                       for pair in puzzle[section]}
        fixed_size_puzzles += len(pair_shapes) == 1
        mixed_size_puzzles += len(pair_shapes) > 1
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        train_bands = {
            (item["col_lengths"], row)
            for item in info["train_infos"] for row in item["markers"]
        }
        if len(neutral_pairs) > 1 and any(
            len(set(item["row_lengths"])) > 1
            or len(set(item["col_lengths"])) > 1
            for item in all_infos
        ) and any(
            row[j] == row[j + 1]
            for item in all_infos
            for row in item["markers"]
            for j in range(len(row) - 1)
        ):
            combined_variations += 1
        marker_counts = {marker: 0 for marker in info["mapping"]}
        for item in info["train_infos"]:
            for marker, _ in item["observations"]:
                marker_counts[marker] += 1
        rare_training_code += min(marker_counts.values()) == 1
        for section in ("train", "test"):
            for pair, item in zip(puzzle[section], info[section + "_infos"]):
                shape = tuple(pair["input"].shape)
                shapes.add(shape)
                row_counts.add(len(item["row_runs"]))
                col_counts.add(len(item["col_runs"]))
                unequal += len(set(item["row_lengths"])) > 1 or len(set(item["col_lengths"])) > 1
                rectangular += any(h != w for h in item["row_lengths"] for w in item["col_lengths"])
                single_column += len(item["col_runs"]) == 1
                wide += len(item["col_runs"]) > 3
                tall += len(item["row_runs"]) > 2
                all_input_keys.add(_grid_signature(pair["input"]))
                predicted = _predict_from_training(
                    [{"input": x["input"].copy(), "output": x["output"].copy()}
                     for x in puzzle["train"]], pair["input"].copy())
                _require(np.array_equal(predicted, pair["output"]),
                         "training-derived prediction mismatch")
                markers = item["markers"]
                if section == "train":
                    train_many_bands += len(item["row_runs"]) > 2
                    train_single_marker += len({x for row in markers for x in row}) == 1
                    train_adjacent += any(row[j] == row[j + 1]
                                          for row in markers for j in range(len(row) - 1))
                    train_three_distinct += any(len(set(row)) >= 3 for row in markers)
                    train_identical_band += any(
                        len(row) >= 3 and len(set(row)) == 1 for row in markers
                    )
                else:
                    test_many_bands += len(item["row_runs"]) > 2
                    new_test_first_band += (
                        (item["col_lengths"], markers[0]) not in train_bands
                    )
                    repeated = any(row[j] == row[j + 1]
                                   for row in markers for j in range(len(row) - 1))
                    test_adjacent += repeated
                    test_nonadjacent += not repeated
                    same_size_test += shape in train_shapes
                    changed_size_test += shape not in train_shapes
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "pair count diversity insufficient")
    _require(code_counts == {2, 3, 4}, "codebook size diversity insufficient")
    _require(len(shapes) >= 25 and len(all_input_keys) >= 200,
             "grid diversity insufficient")
    _require({1, 3, 4}.issubset(row_counts) and {1, 3, 4}.issubset(col_counts),
             "compartment-count diversity insufficient")
    _require(all((stable_neutral, variable_neutral, unequal, rectangular,
                  single_column, wide, tall, train_single_marker,
                  train_adjacent, train_three_distinct,
                  train_identical_band, test_adjacent, test_nonadjacent,
                  same_size_test, changed_size_test, fixed_size_puzzles,
                  mixed_size_puzzles, rare_training_code,
                  train_many_bands, test_many_bands,
                  new_test_first_band, combined_variations)),
             "review variation axis missing")
    _require(_puzzles_equal(generate_puzzle(7), generate_puzzle(7)),
             "generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(1001, 6, 3),
                                generate_puzzle(1002, 6, 3)),
             "different seeds give the same puzzle")
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4)), 2000):
        puzzle = generate_puzzle(seed, *counts)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts,
                 "explicit counts ignored")
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 3000):
        puzzle = generate_puzzle(seed, *counts)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested,
                     "partial count ignored")
    illegal = (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 2.5},
        {"num_test": True},
    )
    for index, arguments in enumerate(illegal):
        try:
            generate_puzzle(4000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {arguments}")
