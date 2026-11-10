import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "0000000",
        "0111110",
        "0022200",
        "0022200",
        "0333330",
        "0003000",
        "0000000",
    ),
    (
        "0000000000000",
        "0001110111000",
        "0000011100000",
        "0002222222000",
        "0002000002000",
        "0002222222000",
        "0000033300000",
        "0000030300000",
        "0000003000000",
        "0444444444440",
        "0005555555000",
        "0005500055000",
        "0000000000000",
    ),
    (
        "0000000000000",
        "0008888888000",
        "0002222222000",
        "0000002000000",
        "0003333333000",
        "0000001000000",
        "0000001000000",
        "0001111111000",
        "0001000001000",
        "0001111111000",
        "0000004000000",
        "0004444444000",
        "0000000000000",
    ),
    (
        "0000000000",
        "0033333300",
        "0030330300",
        "0033333300",
        "0002222000",
        "0002002000",
        "0002222000",
        "0000220000",
        "0044444400",
        "0008008000",
        "0000880000",
        "0006666000",
        "0000000000",
    ),
    (
        "0000000",
        "0444440",
        "0404040",
        "0055500",
        "0606060",
        "0060600",
        "0000000",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _matches_source_input(grid: np.ndarray) -> bool:
    for rows in _SOURCE_INPUT_ROWS:
        if grid.shape != (len(rows), len(rows[0])):
            continue

        rendered = tuple(
            "".join(str(int(value)) for value in row)
            for row in grid
        )
        if rendered == rows:
            return True

    return False


def _analyze_input(
    grid: np.ndarray,
    expected_background: int | None = None,
) -> dict:
    _require(
        isinstance(grid, np.ndarray) and grid.ndim == 2,
        "grid must be a 2-D ndarray",
    )

    background = int(grid[0, 0])

    if expected_background is not None:
        _require(
            background == int(expected_background),
            "inconsistent puzzle background",
        )

    _require(
        bool(np.all(grid[0] == background)),
        "top row must be background",
    )
    _require(
        bool(np.all(grid[-1] == background)),
        "bottom row must be background",
    )

    occupied = np.flatnonzero(
        np.any(grid != background, axis=1)
    )
    _require(
        occupied.size > 0,
        "input must contain objects",
    )

    y0 = int(occupied[0])
    y1 = int(occupied[-1])

    _require(
        y0 >= 1 and y1 <= grid.shape[0] - 2,
        "stack needs outer vertical margins",
    )
    _require(
        np.array_equal(occupied, np.arange(y0, y1 + 1)),
        "active rows must be contiguous",
    )

    row_colors = []

    for y in range(y0, y1 + 1):
        values = np.unique(
            grid[y][grid[y] != background]
        )
        _require(
            values.size == 1,
            "each active row must belong to one monochromatic object",
        )
        row_colors.append(int(values[0]))

    blocks = []
    colors = []
    start = y0
    current = row_colors[0]

    for offset, color in enumerate(
        row_colors[1:],
        start=1,
    ):
        if color != current:
            end = y0 + offset
            blocks.append(grid[start:end].copy())
            colors.append(current)
            start = end
            current = color

    blocks.append(grid[start:y1 + 1].copy())
    colors.append(current)

    _require(
        len(colors) == len(set(colors)),
        "each object must have its own color",
    )

    return {
        "background": background,
        "y0": y0,
        "y1": y1,
        "blocks": blocks,
        "colors": tuple(colors),
    }


def _reverse_object_blocks(
    grid: np.ndarray,
    background: int | None = None,
) -> np.ndarray:
    info = _analyze_input(grid, background)

    output = np.full(
        grid.shape,
        info["background"],
        dtype=grid.dtype,
    )

    cursor = info["y0"]

    for block in reversed(info["blocks"]):
        block_height = block.shape[0]
        output[cursor:cursor + block_height] = block
        cursor += block_height

    _require(
        cursor == info["y1"] + 1,
        "reversed stack must fill the same active span",
    )

    return output


def _normalize_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(
        num_train,
        "num_train",
        2,
        10,
    )
    test_count = _normalize_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(test_count + 1, 11)
        )
    elif train_count is None:
        train_count = int(
            rng.integers(test_count + 1, 11)
        )
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(
            rng.integers(1, maximum_test + 1)
        )

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count



def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The reversal direction is invariant. These parameters only control
    # the visible backgrounds used by the pairs.
    return {
        "anchor_background": int(rng.integers(0, 10)),
        "vary_background": bool(rng.integers(0, 2)),
    }


def _sample_dimensions(
    rng: np.random.Generator,
    regime: str,
    band: str = "any",
) -> tuple[int, int]:
    if band == "small":
        low, high = 7, 10
    elif band == "large":
        low, high = 11, 18
    elif regime == "source_train":
        low, high = 7, 18
    else:
        low, high = 7, 21

    if rng.random() < 0.27:
        familiar = [(7, 7), (13, 13), (13, 10)]
        choices = [(h, w) for h, w in familiar if low <= h <= high]
        if choices:
            return choices[int(rng.integers(0, len(choices)))]

    height = int(rng.integers(low, high + 1))
    square_chance = 0.51 if regime == "source_train" else 0.34
    if rng.random() < square_chance:
        return height, height

    minimum = max(5, int(np.ceil(0.56 * height)))
    maximum = min(23, int(np.floor(1.43 * height)))
    width = int(rng.integers(minimum, maximum + 1))
    if width == height:
        width = width + 1 if width < maximum else width - 1
    return height, width


def _sample_count(
    rng: np.random.Generator,
    height: int,
    at_least_three: bool,
) -> int:
    minimum = 3 if at_least_three or height >= 11 else 2
    if height <= 8:
        maximum = min(4, height - 3)
    elif height <= 14:
        maximum = min(6, height - 3)
    else:
        maximum = min(8, height - 3)
    return int(rng.integers(minimum, maximum + 1))


def _sample_heights(
    rng: np.random.Generator,
    canvas_height: int,
    count: int,
) -> np.ndarray:
    interior = canvas_height - 2
    minimum = max(count + 1, int(np.ceil(0.72 * interior)))
    if rng.random() < 0.13:
        minimum = count + 1
    occupied = int(rng.integers(minimum, interior + 1))
    heights = np.ones(count, dtype=int)
    extra = occupied - count

    # Several objects get actual silhouettes before surplus is distributed.
    recipients = rng.permutation(count)[:min(extra, 3)]
    heights[recipients] += 1
    extra -= len(recipients)
    if extra:
        weights = rng.gamma(1.3, 1.0, size=count)
        if rng.random() < 0.27:
            weights[int(rng.integers(0, count))] *= 5
        heights += rng.multinomial(extra, weights / weights.sum())
    return heights


def _sample_object_colors(
    rng: np.random.Generator,
    background: int,
    count: int,
    color_order: str,
) -> np.ndarray:
    available = np.array([c for c in range(10) if c != background])
    colors = rng.choice(available, size=count, replace=False)
    if color_order == "random":
        color_order = (
            "monotone" if rng.random() < 0.29 else "nonmonotone"
        )
    if color_order == "monotone":
        colors.sort()
        if rng.random() < 0.5:
            colors = colors[::-1]
    elif color_order == "nonmonotone" and count >= 3:
        # A permutation is enough; this fallback makes the guarantee exact.
        rng.shuffle(colors)
        if _monotone(colors):
            colors[[1, 2]] = colors[[2, 1]]
    return colors.astype(np.int8)


def _monotone(colors: np.ndarray | tuple) -> bool:
    steps = np.diff(np.asarray(colors, dtype=int))
    return bool(np.all(steps > 0) or np.all(steps < 0))


def _make_mask(
    rng: np.random.Generator,
    height: int,
    span: int,
    style: int,
    force_asymmetric: bool,
) -> np.ndarray:
    mask = np.zeros((height, span), dtype=bool)
    center = span // 2

    if style == 0:  # filled or gapped bar
        mask[:] = True
        if span >= 5 and rng.random() < 0.4:
            mask[:, int(rng.integers(1, span - 1))] = False
    elif style == 1:  # outline
        mask[0] = True
        mask[-1] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif style == 2:  # cap and stem
        mask[0] = True
        mask[:, center] = True
        if height >= 3 and rng.random() < 0.45:
            mask[-1] = True
    elif style == 3:  # hook
        mask[:, 0] = True
        mask[-1] = True
        if height >= 3 and span >= 3:
            mask[0, : max(2, span // 2)] = True
    elif style == 4:  # split arms, joined in selected rows
        mask[:, 0] = True
        mask[:, -1] = True
        if span >= 3:
            mask[0] = True
            if height >= 3 and rng.random() < 0.5:
                mask[-1] = True
    elif style == 5:  # moving stem
        for y in range(height):
            x = (int(rng.integers(0, span)) + y) % span
            mask[y, x] = True
            if span >= 4 and y % 2 == 0:
                mask[y, (x + 2) % span] = True
    elif style == 6:  # tapered silhouette
        for y in range(height):
            inset = min(y % max(1, (span + 1) // 2), span - 1)
            mask[y, inset:span - inset] = True
            if not mask[y].any():
                mask[y, center] = True
    else:  # irregular, including separated pixels in the same block
        mask = rng.random((height, span)) < float(rng.uniform(0.35, 0.75))

    for y in range(height):
        if not mask[y].any():
            mask[y, int(rng.integers(0, span))] = True

    if force_asymmetric and np.array_equal(mask, mask[::-1]):
        # Change the last row without making it blank.
        x = int(np.flatnonzero(mask[-1])[0])
        if mask[-1].sum() > 1:
            mask[-1, x] = False
        else:
            mask[-1, (x + 1) % span] = True
    return mask


def _sample_span_and_x(
    rng: np.random.Generator,
    width: int,
    minimum_span: int,
    allow_edge: bool,
    force_edge: bool,
) -> tuple[int, int]:
    maximum_span = width if allow_edge else width - 2
    draw = rng.random()
    if draw < 0.12 and minimum_span == 1:
        span = 1
    elif draw < 0.45:
        span = int(rng.integers(max(minimum_span, maximum_span // 2), maximum_span + 1))
    elif draw < 0.68:
        span = int(rng.integers(minimum_span, min(maximum_span, 5) + 1))
    else:
        span = int(rng.integers(minimum_span, maximum_span + 1))

    if force_edge or (allow_edge and rng.random() < 0.18):
        x0 = 0 if rng.random() < 0.5 else width - span
    elif allow_edge:
        x0 = int(rng.integers(0, width - span + 1))
    else:
        x0 = int(rng.integers(1, width - span))
    return span, x0


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    shape: tuple[int, int],
    background: int,
    color_order: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    height, width = shape
    count = _sample_count(rng, height, color_order == "nonmonotone")
    heights = _sample_heights(rng, height, count)
    colors = _sample_object_colors(rng, background, count, color_order)
    occupied = int(heights.sum())
    top_margin = int(rng.integers(1, height - occupied))
    grid = np.full(shape, background, dtype=np.int8)
    witness_index = int(np.argmax(heights))
    allow_edge = bool(rng.random() < 0.36)
    cursor = top_margin

    for index, (block_height, color) in enumerate(zip(heights, colors)):
        witness = index == witness_index and block_height >= 2
        span, x0 = _sample_span_and_x(
            rng, width, 2 if witness else 1, allow_edge,
            allow_edge and index == 0,
        )
        mask = _make_mask(
            rng, int(block_height), span,
            int(rng.integers(0, 8)), witness,
        )
        if x0 == 0:
            mask[0, 0] = True
        if x0 + span == width:
            mask[0, -1] = True
        grid[cursor:cursor + block_height, x0:x0 + span][mask] = color
        cursor += int(block_height)

    return grid, _reverse_object_blocks(grid)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "illegal pair counts")

    input_keys, pair_keys = set(), set()
    train_infos = []
    train_configs = set()
    test_configs = set()
    shapes = set()

    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "invalid pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2-D ndarray")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30,
                         "illegal grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((grid >= 0) & (grid <= 9))),
                         "illegal grid values")
            grid, output = pair["input"], pair["output"]
            _require(grid.shape == output.shape, "shape changed")
            key, pkey = _grid_key(grid), _pair_key(pair)
            _require(key not in input_keys and pkey not in pair_keys,
                     "duplicate generated pair")
            input_keys.add(key)
            pair_keys.add(pkey)
            _require(not _matches_source_input(grid), "source input copy")
            info = _analyze_input(grid)
            _require(2 <= len(info["blocks"]) <= 8, "illegal object count")
            _require(np.array_equal(output, _reverse_object_blocks(grid)),
                     "incorrect object reversal")
            _require(not np.array_equal(grid, output), "unchanged pair")
            out_info = _analyze_input(output, info["background"])
            _require(out_info["colors"] == tuple(reversed(info["colors"])),
                     "object colors not reversed")
            _require(out_info["y0"] == info["y0"]
                     and out_info["y1"] == info["y1"],
                     "occupied span changed")
            config = (grid.shape, len(info["blocks"]))
            shapes.add(grid.shape)
            if split == "train":
                train_infos.append(info)
                train_configs.add(config)
            else:
                test_configs.add(config)

    _require(len(shapes) >= 2, "puzzle has no dimension variation")
    _require(any(config not in train_configs for config in test_configs),
             "test lacks a new shape/count configuration")
    _require(any(
        block.shape[0] >= 2 and not np.array_equal(block, block[::-1])
        for info in train_infos for block in info["blocks"]
    ), "training does not distinguish block reversal from row reversal")
    _require(any(_monotone(info["colors"]) for info in train_infos),
             "training lacks a simple color order")
    _require(any(not _monotone(info["colors"]) for info in train_infos),
             "training does not defeat numeric color sorting")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # Different bands guarantee a visible training range; their order is random.
    train_shapes = [
        _sample_dimensions(rng, "source_train", "small"),
        _sample_dimensions(rng, "source_train", "large"),
    ]
    train_shapes.extend(
        _sample_dimensions(rng, "source_train")
        for _ in range(train_count - 2)
    )
    rng.shuffle(train_shapes)
    tags = ["random"] * train_count
    special = rng.choice(train_count, size=2, replace=False)
    tags[int(special[0])] = "nonmonotone"
    tags[int(special[1])] = "monotone"
    train_specs = list(zip(train_shapes, tags))

    test_specs = [
        (_sample_dimensions(rng, "target_test"), "random")
        for _ in range(test_count)
    ]
    shifted_index = int(rng.integers(0, test_count))
    train_shape_set = set(train_shapes)
    if test_specs[shifted_index][0] in train_shape_set:
        for _ in range(40):
            shape = _sample_dimensions(rng, "target_test")
            if shape not in train_shape_set:
                test_specs[shifted_index] = (shape, "random")
                break
        else:
            raise AssertionError("could not construct a new test shape")

    puzzle = {"train": [], "test": []}
    seen = set()
    pair_number = 0
    for regime, specs, split in (
        ("source_train", train_specs, "train"),
        ("target_test", test_specs, "test"),
    ):
        for shape, tag in specs:
            if not parameters["vary_background"] or pair_number == 0:
                background = parameters["anchor_background"]
            elif pair_number == 1:
                available = [c for c in range(10)
                             if c != parameters["anchor_background"]]
                background = int(rng.choice(available))
            else:
                background = int(rng.integers(0, 10))
            for _ in range(40):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**63 - 1))
                )
                grid, output = generate_pair(
                    pair_rng, regime, shape, background, tag
                )
                key = _grid_key(grid)
                if key not in seen and not _matches_source_input(grid):
                    seen.add(key)
                    puzzle[split].append({"input": grid, "output": output})
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            pair_number += 1

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, heights, counts = set(), set(), set()
    conditional = {7: set(), 13: set()}
    backgrounds = set()
    variable_background_puzzles = 0
    constant_background_puzzles = 0
    monotone_train = nonmonotone_train = 0
    edge_pairs = one_row_blocks = tall_blocks = 0
    layouts = set()
    all_keys = []
    combination_pairs = 0

    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bgs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                bgs.add(info["background"])
                backgrounds.add(info["background"])
                h, w = grid.shape
                shapes.add((h, w))
                heights.add(h)
                count = len(info["blocks"])
                counts.add(count)
                if h in conditional:
                    conditional[h].add(count)
                if split == "train":
                    if _monotone(info["colors"]):
                        monotone_train += 1
                    else:
                        nonmonotone_train += 1
                block_heights = tuple(block.shape[0] for block in info["blocks"])
                one_row_blocks += sum(v == 1 for v in block_heights)
                tall_blocks += sum(v >= 3 for v in block_heights)
                edge = bool(np.any(grid[:, 0] != info["background"])
                            or np.any(grid[:, -1] != info["background"]))
                edge_pairs += edge
                combination_pairs += edge and any(v >= 3 for v in block_heights) and h != w
                layouts.add((grid.shape, info["y0"], block_heights,
                             tuple((int(np.count_nonzero(block)),
                                    int(np.flatnonzero(block != info["background"])[0] % w))
                                   for block in info["blocks"])))
                all_keys.append(_grid_key(grid))
        variable_background_puzzles += len(bgs) > 1
        constant_background_puzzles += len(bgs) == 1

    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(shapes) >= 20 and len(heights) >= 9,
             "canvas sizes lack diversity")
    _require(len(counts) >= 5 and len(conditional[7]) >= 2
             and len(conditional[13]) >= 2,
             "object counts lack conditional diversity")
    _require(len(backgrounds) >= 8 and variable_background_puzzles >= 8
             and constant_background_puzzles >= 8,
             "background roles lack diversity")
    _require(monotone_train >= 20 and nonmonotone_train >= 20,
             "training color orders lack diversity")
    _require(edge_pairs >= 15 and one_row_blocks >= 10 and tall_blocks >= 30
             and combination_pairs >= 5,
             "object geometry lacks diversity")
    _require(len(layouts) >= 100 and len(set(all_keys)) >= 0.95 * len(all_keys),
             "structural layouts repeat too often")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40,
             "different seeds insufficiently differ")
    _require(_puzzle_signature(generate_puzzle(777, 6, 3))
             == _puzzle_signature(generate_puzzle(777, 6, 3)),
             "same-call determinism failed")

    legal = ((2, 1), (10, 1), (10, 4), (None, 1),
             (None, 4), (2, None), (10, None))
    for offset, (nt, ns) in enumerate(legal):
        puzzle = generate_puzzle(2000 + offset, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "train count ignored")
        if ns is not None:
            _require(len(puzzle["test"]) == ns, "test count ignored")

    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
