import numpy as np


_SOURCE_PAIR_ROW_STRINGS = frozenset(
    (
        (
            (
                "202200020022",
                "201202080202",
                "122228200020",
                "200022202200",
                "221022282022",
                "200202022200",
            ),
            (
                "2020",
                "2222",
                "0222",
            ),
        ),
        (
            (
                "20000200022",
                "22002002020",
                "21020000280",
                "00000000200",
            ),
            (
                "0200002",
            ),
        ),
        (
            (
                "2200220220",
                "1020802000",
                "2200020002",
                "2020012008",
                "2200220220",
            ),
            (
                "020",
                "200",
            ),
        ),
        (
            (
                "120200082002",
                "120202080002",
                "102202280022",
                "220000220022",
                "022000002220",
                "022220000022",
                "020200202022",
                "200200222000",
            ),
            (
                "202000",
                "202020",
                "022022",
            ),
        ),
        (
            (
                "22202202202",
                "20000222000",
                "02222022022",
                "20122200820",
                "00200202000",
                "12200280002",
                "00000020222",
                "00212000280",
                "02001222028",
                "02022022200",
                "20020002020",
                "21222228220",
                "22202002002",
                "00202022220",
            ),
            (
                "22200",
                "22002",
                "20002",
                "22202",
                "22222",
            ),
        ),
        (
            (
                "20102022820000",
                "22200002200002",
                "01022028200002",
                "02100000822022",
                "20202022022200",
                "01220008022222",
                "20002200202220",
                "22221002008022",
                "00002002200022",
            ),
            (
                "02022",
                "02202",
                "00000",
                "22000",
                "00200",
            ),
        ),
    )
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    normalized = int(value)
    if not lower <= normalized <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return normalized


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This mapping gives deterministic variation and guarantees broad
        # count coverage for consecutive seeds.
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % span
        return train_count, test_count

    if train_count is None:
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + seed % span
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    test_count = 1 + seed % maximum_test
    return train_count, test_count


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    palette_size = int(rng.choice([1, 2, 2, 3, 3, 4]))
    colors = rng.choice(10, size=palette_size + 2, replace=False)
    return {
        "payload": tuple(int(x) for x in colors[:palette_size]),
        "opening": int(colors[-2]),
        "closing": int(colors[-1]),
    }


def _sample_rows(rng, height, count, mode):
    if mode == "all":
        return np.arange(height, dtype=int)
    if mode == "consecutive":
        start = int(rng.integers(0, height - count + 1))
        return np.arange(start, start + count, dtype=int)
    if mode == "clustered":
        span = min(height, count + int(rng.integers(1, 5)))
        start = int(rng.integers(0, height - span + 1))
        return np.sort(rng.choice(np.arange(start, start + span), count, replace=False))
    if mode == "isolated" and height >= 2 * count - 1:
        # Choose nonadjacent rows by choosing their compressed coordinates.
        compressed = np.sort(rng.choice(height - count + 1, count, replace=False))
        return compressed + np.arange(count)
    return np.sort(rng.choice(height, count, replace=False))


def _sample_canvas(rng, regime, count, segment_width):
    row_mode = str(rng.choice(["all", "consecutive", "clustered", "scattered", "isolated"],
                              p=[0.13, 0.20, 0.20, 0.29, 0.18]))
    cap = 22 if regime == "source_train" else 27
    if row_mode == "all":
        height = count
    else:
        extra = int(rng.integers(1, min(15, cap - count) + 1))
        height = count + extra
    width_cap = 24 if regime == "source_train" else 28
    minimum_width = segment_width + 2
    # A wide tail and a narrow tail both occur in each regime.
    if rng.random() < 0.35:
        width = int(rng.integers(minimum_width, min(width_cap, minimum_width + 5) + 1))
    else:
        width = int(rng.integers(minimum_width, width_cap + 1))
    return height, width, row_mode


def _sample_starts(rng, width, segment_width, count):
    last_start = width - segment_width - 2
    if rng.random() < 0.23:
        start = int(rng.integers(0, last_start + 1))
        return np.full(count, start, dtype=int)
    starts = rng.integers(0, last_start + 1, size=count).astype(int)
    if last_start and rng.random() < 0.24:
        starts[int(rng.integers(0, count))] = int(rng.choice([0, last_start]))
    return starts


def _choose_pair_palette(rng, puzzle_palette, cover_all):
    colors = np.asarray(puzzle_palette, dtype=np.int8)
    if cover_all or len(colors) == 1:
        return colors
    if rng.random() < 0.55:
        return colors
    size = int(rng.integers(1, len(colors)))
    return rng.choice(colors, size=size, replace=False)


def _color_weights(rng, colors, skewed):
    if len(colors) == 1:
        return np.ones(1)
    if skewed:
        dominant = int(rng.integers(0, len(colors)))
        weight = float(rng.uniform(0.72, 0.96))
        probabilities = np.full(len(colors), (1.0 - weight) / (len(colors) - 1))
        probabilities[dominant] = weight
        return probabilities
    weights = rng.uniform(0.65, 1.35, size=len(colors))
    return weights / weights.sum()


def _sample_texture(rng, shape, colors, mode):
    height, width = shape
    if mode == "solid":
        return np.full(shape, rng.choice(colors), dtype=np.int8)
    if mode == "runs":
        result = np.empty(shape, dtype=np.int8)
        for row in range(height):
            column = 0
            while column < width:
                length = int(rng.integers(1, min(width - column, max(2, width // 2)) + 1))
                result[row, column:column + length] = rng.choice(colors)
                column += length
        return result
    if mode == "patches":
        row_scale = int(rng.integers(2, 5))
        col_scale = int(rng.integers(2, 5))
        coarse = rng.choice(colors, size=((height + row_scale - 1) // row_scale,
                                          (width + col_scale - 1) // col_scale))
        return np.repeat(np.repeat(coarse, row_scale, axis=0), col_scale, axis=1)[:height, :width].astype(np.int8)
    if mode == "solid_rows":
        row_colors = rng.choice(colors, size=height)
        return np.repeat(row_colors[:, None], width, axis=1).astype(np.int8)
    probabilities = _color_weights(rng, colors, skewed=(mode == "skewed"))
    return rng.choice(colors, size=shape, p=probabilities).astype(np.int8)


def _generate_pair(rng, regime, parameters, marked_count, segment_width, cover_all=False):
    height, width, row_mode = _sample_canvas(rng, regime, marked_count, segment_width)
    marked_rows = _sample_rows(rng, height, marked_count, row_mode)
    starts = _sample_starts(rng, width, segment_width, marked_count)
    palette = _choose_pair_palette(rng, parameters["payload"], cover_all)

    field_mode = str(rng.choice(["random", "skewed", "runs", "patches", "solid"],
                                p=[0.34, 0.21, 0.19, 0.20, 0.06]))
    output_mode = str(rng.choice(["random", "skewed", "runs", "patches", "solid_rows", "solid"],
                                 p=[0.30, 0.17, 0.18, 0.16, 0.12, 0.07]))
    field = _sample_texture(rng, (height, width), palette, field_mode)
    block = _sample_texture(rng, (marked_count, segment_width), palette, output_mode)
    if cover_all:
        # Training exposes every payload color even if this pair was sampled with a skewed texture.
        positions = rng.choice(block.size, size=len(palette), replace=False)
        for position, color in zip(positions, palette):
            block.flat[int(position)] = color

    # Copying a plausible payload into an unmarked row makes the markers decisive.
    unmarked = np.setdiff1d(np.arange(height), marked_rows)
    if len(unmarked) and rng.random() < 0.55:
        for row in rng.choice(unmarked, size=int(rng.integers(1, len(unmarked) + 1)), replace=False):
            start = int(rng.integers(0, width - segment_width + 1))
            field[int(row), start:start + segment_width] = block[int(rng.integers(0, marked_count))]

    opening, closing = parameters["opening"], parameters["closing"]
    for output_row, (row, start) in enumerate(zip(marked_rows, starts)):
        r, s = int(row), int(start)
        field[r, s] = opening
        field[r, s + segment_width + 1] = closing
        field[r, s + 1:s + segment_width + 1] = block[output_row]

    output = _extract_with_known_delimiters(field, opening, closing)
    return field, output


def _grid_signature(grid: np.ndarray) -> tuple:
    return (tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.ravel()))


def _pair_signature(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_signature(input_grid), _grid_signature(output_grid))


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple(_pair_signature(p["input"], p["output"])
                               for p in puzzle[split])) for split in ("train", "test"))


def _rows_as_strings(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return (_rows_as_strings(input_grid), _rows_as_strings(output_grid)) in _SOURCE_PAIR_ROW_STRINGS


def _training_specs(rng, count):
    first_count = int(rng.integers(2, 6))
    first_width = int(rng.integers(4, 8))
    other_count = int(rng.choice([x for x in range(1, 9) if x != first_count]))
    other_width = int(rng.choice([x for x in range(2, 11) if x != first_width]))
    specs = [(first_count, first_width, True), (other_count, other_width, False)]
    specs.extend((int(rng.integers(1, 9)), int(rng.integers(1, 11)), False)
                 for _ in range(count - 2))
    rng.shuffle(specs)
    return specs


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_specs = _training_specs(rng, train_count)
    # The source's 5-by-5 case stays possible; count and width remain
    # independent and span much smaller and larger outputs.
    test_specs = [(int(rng.choice([1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10])),
                   int(rng.choice([1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12])),
                   False) for _ in range(test_count)]
    for split, regime, specs in (("train", "source_train", train_specs),
                                 ("test", "target_test", test_specs)):
        for marked_count, segment_width, cover_all in specs:
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                input_grid, output_grid = _generate_pair(pair_rng, regime, parameters,
                                                         marked_count, segment_width, cover_all)
                signature = _grid_signature(input_grid)
                if signature not in seen_inputs and not _is_source_pair(input_grid, output_grid):
                    seen_inputs.add(signature)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _check(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _check(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _check(1 <= height <= 30 and 1 <= width <= 30, f"{label} has illegal dimensions")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must use integer colors")
    _check(int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9, f"{label} has illegal colors")


def _infer_training_conventions(train_pairs: list) -> dict:
    payload = frozenset(int(x) for pair in train_pairs for x in np.unique(pair["output"]))
    full = frozenset(int(x) for pair in train_pairs for x in np.unique(pair["input"]))
    markers = full - payload
    _check(1 <= len(payload) <= 8 and len(markers) == 2,
           "training must reveal exactly two marker colors and its payload palette")
    orders = set()
    for pair in train_pairs:
        for row in pair["input"]:
            positions = [(column, int(value)) for column, value in enumerate(row) if int(value) in markers]
            if not positions:
                continue
            _check(len(positions) == 2 and {value for _, value in positions} == markers,
                   "each marked row must contain one of each delimiter")
            orders.add((positions[0][1], positions[1][1]))
    _check(len(orders) == 1, "training must show one consistent left-to-right delimiter order")
    opening, closing = next(iter(orders))
    return {"payload": payload, "full_palette": full, "opening": opening, "closing": closing}


def _extract_with_known_delimiters(input_grid: np.ndarray, opening: int, closing: int) -> np.ndarray:
    segments = []
    for row in input_grid:
        left = np.flatnonzero(row == opening)
        right = np.flatnonzero(row == closing)
        if left.size == 0 and right.size == 0:
            continue
        _check(left.size == 1 and right.size == 1, "marked row must have exactly one delimiter of each role")
        a, b = int(left[0]), int(right[0])
        _check(b >= a + 2, "opening must precede closing with a nonempty interior")
        segments.append(row[a + 1:b].copy())
    _check(bool(segments), "each pair must have a marked row")
    _check(len({len(row) for row in segments}) == 1, "interior widths must match within a pair")
    return np.stack(segments, axis=0)


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray, conventions: dict) -> dict:
    opening, closing = conventions["opening"], conventions["closing"]
    payload, full = conventions["payload"], conventions["full_palette"]
    _check(set(int(x) for x in np.unique(input_grid)) <= full, "input uses an unlearned color")
    _check(set(int(x) for x in np.unique(output_grid)) <= payload, "output uses an unlearned color")
    _check(set(int(x) for x in np.unique(input_grid)) - {opening, closing} <= payload,
           "non-marker input colors must be payload colors")
    expected = _extract_with_known_delimiters(input_grid, opening, closing)
    _check(np.array_equal(expected, output_grid), "output must equal ordered between-delimiter extraction")
    rows = []
    starts = []
    for index, row in enumerate(input_grid):
        if np.any(row == opening):
            rows.append(index)
            starts.append(int(np.flatnonzero(row == opening)[0]))
    return {"marked_rows": tuple(rows), "starts": tuple(starts),
            "segment_width": output_grid.shape[1]}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{split}[{index}].input")
            _validate_grid(output_grid, f"{split}[{index}].output")
            input_signature = _grid_signature(input_grid)
            pair_signature = _pair_signature(input_grid, output_grid)
            _check(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                   "duplicate pair or input")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            _check(not _is_source_pair(input_grid, output_grid), "exact source pair copied")
            _check(not np.array_equal(input_grid, output_grid), "input cannot equal output")
    conventions = _infer_training_conventions(puzzle["train"])
    train_heights, train_widths = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair["input"], pair["output"], conventions)
            _check(pair["input"].shape[1] >= pair["output"].shape[1] + 2,
                   "input must fit both delimiters")
            _check(pair["input"].shape[0] >= pair["output"].shape[0],
                   "output cannot have more rows than input")
            if split == "train":
                train_heights.add(pair["output"].shape[0])
                train_widths.add(pair["output"].shape[1])
    _check(len(train_heights) >= 2 and len(train_widths) >= 2,
           "training must vary both extracted row count and width")


def _solve_from_training(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    # This prediction path sees only demonstrations and a test input.
    conventions = _infer_training_conventions(train_pairs)
    return _extract_with_known_delimiters(test_input, conventions["opening"], conventions["closing"])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    palettes, input_shapes, output_shapes, test_shapes, layouts = set(), set(), set(), set(), set()
    all_marked, aligned, shifted, consecutive, gapped, uniform = 0, 0, 0, 0, 0, 0
    palette_sizes, test_row_counts, test_widths = set(), set(), set()
    palette_variation, width_overlap, new_test_shape, first_shape_match = 0, 0, 0, 0
    small_tests, large_tests, familiar_tests, binary_outputs, many_color_outputs = 0, 0, 0, 0, 0
    skewed_outputs, unmarked_mimics, exterior_mimics = 0, 0, 0
    test_aspects = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        conventions = _infer_training_conventions(puzzle["train"])
        palettes.add((tuple(sorted(conventions["payload"])), conventions["opening"], conventions["closing"]))
        palette_sizes.add(len(conventions["payload"]))
        markers = {conventions["opening"], conventions["closing"]}
        pair_palettes = {frozenset(int(x) for x in np.unique(p["input"]) if int(x) not in markers)
                         for split in ("train", "test") for p in puzzle[split]}
        palette_variation += len(pair_palettes) > 1
        width_overlap += any(p["input"].shape[1] <= max(t["input"].shape[1] for t in puzzle["train"])
                             for p in puzzle["test"])
        train_shapes = {p["output"].shape for p in puzzle["train"]}
        new_test_shape += any(p["output"].shape not in train_shapes for p in puzzle["test"])
        first_shape_match += any(p["output"].shape == puzzle["train"][0]["output"].shape
                                 for p in puzzle["test"])
        _check(len({p["output"].shape[0] for p in puzzle["train"]}) >= 2, "train row counts collapsed")
        _check(len({p["output"].shape[1] for p in puzzle["train"]}) >= 2, "train widths collapsed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                info = _analyze_pair(input_grid, output_grid, conventions)
                input_shapes.add(input_grid.shape)
                output_shapes.add(output_grid.shape)
                layouts.add((input_grid.shape, output_grid.shape, info["marked_rows"], info["starts"]))
                rows, starts = info["marked_rows"], info["starts"]
                all_marked += len(rows) == input_grid.shape[0]
                aligned += len(starts) >= 2 and len(set(starts)) == 1
                shifted += len(set(starts)) >= 2
                consecutive += len(rows) >= 2 and any(b == a + 1 for a, b in zip(rows, rows[1:]))
                gapped += len(rows) >= 2 and rows[-1] - rows[0] > len(rows) - 1
                colors, counts = np.unique(output_grid, return_counts=True)
                uniform += len(colors) == 1
                binary_outputs += len(colors) == 2
                many_color_outputs += len(colors) >= 3
                skewed_outputs += int(np.max(counts)) / output_grid.size >= 0.8
                segment_width = output_grid.shape[1]
                for row_index, row in enumerate(input_grid):
                    if row_index not in rows and any(
                        np.array_equal(row[column:column + segment_width], candidate)
                        for column in range(input_grid.shape[1] - segment_width + 1)
                        for candidate in output_grid
                    ):
                        unmarked_mimics += 1
                        break
                for row_index, start in zip(rows, starts):
                    row = input_grid[row_index]
                    for outside in (row[:start], row[start + segment_width + 2:]):
                        if any(np.array_equal(outside[column:column + segment_width], candidate)
                               for column in range(max(0, len(outside) - segment_width + 1))
                               for candidate in output_grid):
                            exterior_mimics += 1
                            break
                if split == "test":
                    test_shapes.add(output_grid.shape)
                    test_row_counts.add(output_grid.shape[0])
                    test_widths.add(output_grid.shape[1])
                    small_tests += output_grid.shape[0] <= 2 and output_grid.shape[1] <= 3
                    large_tests += output_grid.shape[0] >= 8 and output_grid.shape[1] >= 9
                    familiar_tests += output_grid.shape == (5, 5)
                    test_aspects.add("tall" if input_grid.shape[0] > input_grid.shape[1]
                                     else "square" if input_grid.shape[0] == input_grid.shape[1] else "wide")
                    _check(np.array_equal(_solve_from_training(puzzle["train"], input_grid), output_grid),
                           "training-only prediction failed")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic pair counts lack variation")
    _check(len(palettes) >= 30 and len(input_shapes) >= 40 and len(output_shapes) >= 30,
           "palette or dimensional diversity is too narrow")
    _check(len(test_shapes) >= 20 and len(layouts) >= 150, "test shapes or layouts lack diversity")
    _check(palette_sizes == {1, 2, 3, 4} and palette_variation >= 15,
           "seed and pair payload palettes lack variation")
    _check(min(test_row_counts) == 1 and max(test_row_counts) == 10 and
           min(test_widths) == 1 and max(test_widths) == 12,
           "test extraction sizes do not cover the sampled range")
    _check(small_tests > 0 and large_tests > 0 and familiar_tests > 0 and
           width_overlap > 0 and new_test_shape > 0 and first_shape_match < 42,
           "test sizes or training order have collapsed")
    _check(binary_outputs > 0 and many_color_outputs > 0 and skewed_outputs > 0 and
           unmarked_mimics > 0 and exterior_mimics > 0,
           "copied strings or ignored field lack texture variation")
    _check(all_marked > 0 and aligned > 0 and shifted > 0 and consecutive > 0 and gapped > 0
           and uniform > 0 and test_aspects == {"tall", "square", "wide"},
           "reviewed layout, texture, or aspect modes are missing")
    _check(_puzzle_signature(generate_puzzle(314159, 7, 3)) ==
           _puzzle_signature(generate_puzzle(314159, 7, 3)), "same seed is not deterministic")
    _check(len({_puzzle_signature(generate_puzzle(9000 + i, 6, 2)) for i in range(12)}) >= 10,
           "different seeds produced too few puzzles")
    for seed, ntrain, ntest in ((101, 2, 1), (102, 10, 1), (103, 10, 4)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest), "explicit counts ignored")
    for seed, ntrain, ntest in ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        _check((ntrain is None or len(puzzle["train"]) == ntrain) and
               (ntest is None or len(puzzle["test"]) == ntest), "partial counts ignored")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                    {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {request}")
