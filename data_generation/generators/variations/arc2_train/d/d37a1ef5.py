import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00000000000",
        "02222222200",
        "02000000200",
        "02000000200",
        "02050000200",
        "02005000200",
        "02000000200",
        "02000000200",
        "02222222200",
        "00000000000",
        "00000000000",
        "00000000000",
    ),
    (
        "00000000000",
        "02222222200",
        "02000000200",
        "02050050200",
        "02000500200",
        "02000000200",
        "02222222200",
        "00000000000",
    ),
    (
        "000000000000",
        "022222222220",
        "020000000020",
        "020000000020",
        "020050000020",
        "020000500020",
        "020050050020",
        "020000000020",
        "022222222220",
        "000000000000",
        "000000000000",
        "000000000000",
    ),
    (
        "000000000000",
        "002222222220",
        "002000000020",
        "002050000020",
        "002000050020",
        "002000500020",
        "002005000020",
        "002000000020",
        "002000000020",
        "002222222220",
        "000000000000",
        "000000000000",
        "000000000000",
    ),
)


def _source_rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(character) for row in rows for character in row),
    )


_SOURCE_INPUT_KEYS = frozenset(
    _source_rows_key(rows) for rows in _SOURCE_INPUT_ROWS
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(dimension) for dimension in grid.shape),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


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

    if train_count is None and test_count is None:
        test_count = 1 + (seed % 4)
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + ((seed // 4) % span)

    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + (seed % span)

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + (seed % maximum_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    return {
        "colors": colors,
        "vary_palette": bool(rng.random() < 0.58),
        "two_accents": bool(rng.random() < 0.28),
        "chain": bool(rng.random() < 0.16),
    }


def _pair_colors(rng: np.random.Generator, parameters: dict, index: int) -> tuple:
    base = parameters["colors"]
    if not parameters["vary_palette"] or index == 0:
        roles = base[:3]
    elif index == 1:
        roles = (base[1], base[0], base[2])
    elif index == 2:
        roles = (base[0], base[2], base[1])
    else:
        order = rng.permutation(3)
        roles = tuple(base[int(i)] for i in order)
    return roles + ((base[3],) if parameters["two_accents"] else ())


def _frame_shape(rng: np.random.Generator, regime: str, first_train: bool) -> tuple[int, int]:
    if first_train:
        return int(rng.integers(6, 11)), int(rng.integers(6, 12))
    if rng.random() < (0.18 if regime == "source_train" else 0.22):
        side = int(rng.integers(4, 22))
        return side, side
    styles = ("regular", "small", "thin", "large", "general")
    probabilities = (0.35, 0.13, 0.16, 0.13, 0.23) if regime == "source_train" else (0.24, 0.11, 0.23, 0.24, 0.18)
    style = styles[int(rng.choice(5, p=probabilities))]
    if style == "regular":
        return int(rng.integers(5, 13)), int(rng.integers(5, 14))
    if style == "small":
        return int(rng.integers(3, 7)), int(rng.integers(3, 7))
    if style == "thin":
        long = int(rng.integers(10, 25))
        short = int(rng.integers(3, 6))
        return (long, short) if rng.random() < 0.5 else (short, long)
    if style == "large":
        return int(rng.integers(13, 25)), int(rng.integers(12, 25))
    return int(rng.integers(3, 21)), int(rng.integers(3, 21))


def _axis_margins(rng: np.random.Generator, side: int, first_train: bool) -> tuple[int, int]:
    if first_train:
        before = int(rng.integers(1, min(4, 30 - side)))
        after = int(rng.integers(1, min(5, 31 - side - before)))
        return before, after
    room = 30 - side
    maximum = min(6, room)
    def one(limit: int) -> int:
        if limit == 0:
            return 0
        if rng.random() < 0.11:
            return 0
        return int(rng.integers(1, limit + 1))
    before = one(maximum)
    after = one(min(maximum, room - before))
    return before, after


def _sample_accent_positions(
    rng: np.random.Generator,
    inner_height: int,
    inner_width: int,
    style: str,
    force_two: bool,
) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(inner_height) for c in range(inner_width)]
    area = len(positions)
    if style == "chain" and inner_height >= 3 and inner_width >= 3:
        if rng.random() < 0.5:
            r = int(rng.integers(1, inner_height - 1))
            chosen = [(r, c) for c in range(inner_width)]
        else:
            c = int(rng.integers(1, inner_width - 1))
            chosen = [(r, c) for r in range(inner_height)]
    elif style == "single" and not force_two:
        chosen = [positions[int(rng.integers(area))]]
    elif style in ("row", "column", "adjacent") and area >= 3:
        if style == "row" and inner_width >= 2:
            r = int(rng.integers(inner_height))
            columns = rng.choice(inner_width, size=2, replace=False)
            chosen = [(r, int(c)) for c in columns]
        elif style == "column" and inner_height >= 2:
            c = int(rng.integers(inner_width))
            rows = rng.choice(inner_height, size=2, replace=False)
            chosen = [(int(r), c) for r in rows]
        else:
            edges = [(r, c, r + dr, c + dc)
                     for r, c in positions for dr, dc in ((1, 0), (0, 1))
                     if r + dr < inner_height and c + dc < inner_width]
            if edges:
                a, b, d, e = edges[int(rng.integers(len(edges)))]
                chosen = [(a, b), (d, e)]
            else:
                chosen = [positions[int(i)] for i in rng.choice(area, size=2, replace=False)]
        if force_two and len(chosen) < 2:
            chosen = [positions[int(i)] for i in rng.choice(area, size=2, replace=False)]
    elif style == "near":
        border = [(r, c) for r, c in positions
                  if r in (0, inner_height - 1) or c in (0, inner_width - 1)]
        n = min(area - 1, max(2 if force_two else 1, int(rng.integers(1, 5))))
        chosen = [border[int(i)] for i in rng.choice(len(border), size=min(n, len(border)), replace=False)]
    else:
        if style == "dense":
            lower = max(2 if force_two else 1, int(np.ceil(area * 0.16)))
            upper = min(area - 1, max(lower, int(np.ceil(area * 0.38))))
        else:
            lower = 2 if force_two else 1
            upper = min(area - 1, max(lower, min(8, int(np.ceil(area * 0.16)))))
        n = int(rng.integers(lower, upper + 1))
        chosen = [positions[int(i)] for i in rng.choice(area, size=n, replace=False)]
    # A complete chain leaves background on both sides; all other styles
    # must also leave at least one cell to fill.
    if len(chosen) >= area:
        chosen = chosen[:-1]
    if force_two and len(chosen) < 2:
        chosen = [positions[int(i)] for i in rng.choice(area, size=2, replace=False)]
    return chosen


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    index: int,
    chain_demo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    first_train = regime == "source_train" and index == 0
    frame_height, frame_width = _frame_shape(rng, regime, first_train)
    if chain_demo:
        frame_height = max(5, frame_height)
        frame_width = max(5, frame_width)
    # A 3 by 3 frame has only one interior cell, leaving nothing to fill.
    if frame_height == frame_width == 3:
        frame_width = 4
    top, bottom_margin = _axis_margins(rng, frame_height, first_train)
    left, right_margin = _axis_margins(rng, frame_width, first_train)
    if top == bottom_margin == left == right_margin == 0:
        bottom_margin = 1
    height = top + frame_height + bottom_margin
    width = left + frame_width + right_margin
    bottom = top + frame_height - 1
    right = left + frame_width - 1
    if regime == "target_test" and parameters["vary_palette"]:
        palette = parameters["observed_palettes"][int(rng.integers(len(parameters["observed_palettes"])))]
    else:
        palette = _pair_colors(rng, parameters, index)
    background, frame, *accents = palette
    grid = np.full((height, width), background, dtype=np.int8)
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top:bottom + 1, left] = frame
    grid[top:bottom + 1, right] = frame
    inner_height, inner_width = frame_height - 2, frame_width - 2
    if chain_demo and inner_height >= 3 and inner_width >= 3:
        style = "chain"
    else:
        styles = ("single", "row", "column", "adjacent", "near", "sparse", "dense")
        weights = (0.15, 0.12, 0.11, 0.15, 0.12, 0.23, 0.12)
        style = styles[int(rng.choice(len(styles), p=weights))]
    force_two = parameters["two_accents"] and first_train
    chosen = _sample_accent_positions(rng, inner_height, inner_width, style, force_two)
    for accent_index, (r, c) in enumerate(chosen):
        color = accents[accent_index % len(accents)] if parameters["two_accents"] else accents[0]
        grid[top + 1 + r, left + 1 + c] = color
    output = grid.copy()
    interior = output[top + 1:bottom, left + 1:right]
    interior[interior == background] = frame
    return grid, output


def _assert_valid_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= d <= 30 for d in grid.shape)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def _inspect_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert np.any(changed)
    from_colors = np.unique(input_grid[changed])
    to_colors = np.unique(output_grid[changed])
    assert len(from_colors) == len(to_colors) == 1
    background, frame = int(from_colors[0]), int(to_colors[0])
    assert background != frame
    rows, columns = np.where(input_grid == frame)
    assert len(rows) > 0
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(columns.min()), int(columns.max())
    assert bottom - top >= 2 and right - left >= 2
    frame_mask = np.zeros(input_grid.shape, dtype=bool)
    frame_mask[top, left:right + 1] = True
    frame_mask[bottom, left:right + 1] = True
    frame_mask[top:bottom + 1, left] = True
    frame_mask[top:bottom + 1, right] = True
    assert np.array_equal(input_grid == frame, frame_mask)
    inside = np.zeros(input_grid.shape, dtype=bool)
    inside[top + 1:bottom, left + 1:right] = True
    exterior = ~(inside | frame_mask)
    assert np.any(exterior) and np.all(input_grid[exterior] == background)
    accent_mask = inside & (input_grid != background)
    accent_colors = tuple(sorted(int(x) for x in np.unique(input_grid[accent_mask])))
    assert 1 <= len(accent_colors) <= 2
    assert frame not in accent_colors and background not in accent_colors
    assert np.all(input_grid[inside & ~accent_mask] == background)
    assert np.any(inside & ~accent_mask)
    expected = input_grid.copy()
    expected[inside & ~accent_mask] = frame
    assert np.array_equal(output_grid, expected)
    accent_rows, accent_columns = np.where(accent_mask)
    relatives = tuple(sorted((int(r) - top - 1, int(c) - left - 1)
                             for r, c in zip(accent_rows, accent_columns)))
    inner_height, inner_width = bottom - top - 1, right - left - 1
    accent_set = set(relatives)
    chain = (inner_height >= 3 and inner_width >= 3 and
             (any(all((r, c) in accent_set for c in range(inner_width)) for r in range(1, inner_height - 1)) or
              any(all((r, c) in accent_set for r in range(inner_height)) for c in range(1, inner_width - 1))))
    adjacent = any((r + 1, c) in accent_set or (r, c + 1) in accent_set for r, c in relatives)
    near = any(r in (0, inner_height - 1) or c in (0, inner_width - 1) for r, c in relatives)
    return {
        "background": background, "frame": frame, "accents": accent_colors,
        "top_margin": top, "bottom_margin": input_grid.shape[0] - bottom - 1,
        "left_margin": left, "right_margin": input_grid.shape[1] - right - 1,
        "frame_height": bottom - top + 1, "frame_width": right - left + 1,
        "accent_count": len(relatives), "accent_positions": relatives,
        "changed_count": int(changed.sum()), "chain": chain,
        "adjacent": adjacent, "near": near,
    }


def _validate_regime(info: dict, regime: str) -> None:
    assert regime in ("source_train", "target_test")
    assert info["frame_height"] >= 3 and info["frame_width"] >= 3
    assert 1 <= info["accent_count"]
    assert info["changed_count"] >= 1
    assert all(info[name] >= 0 for name in ("top_margin", "bottom_margin", "left_margin", "right_margin"))
    assert any(info[name] > 0 for name in ("top_margin", "bottom_margin", "left_margin", "right_margin"))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for section, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _assert_valid_grid(inp)
            _assert_valid_grid(out)
            assert inp.dtype == out.dtype
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in inputs and (input_key, output_key) not in pairs
            assert input_key not in _SOURCE_INPUT_KEYS
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            info = _inspect_pair(inp, out)
            _validate_regime(info, regime)
            infos[section].append(info)
    train, test = infos["train"], infos["test"]
    assert all(train[0][name] > 0 for name in ("top_margin", "bottom_margin", "left_margin", "right_margin"))
    assert len({(x["frame_height"], x["frame_width"]) for x in train}) >= 2
    assert all(any(x["background"] == y["background"] and x["frame"] == y["frame"] and
                   set(x["accents"]) <= set(y["accents"]) for y in train) for x in test)
    train_structural_roles = {(x["background"], x["frame"]) for x in train}
    if len(train_structural_roles) == 1:
        demonstrated_accents = set().union(*(set(x["accents"]) for x in train))
        assert all((x["background"], x["frame"]) in train_structural_roles and
                   set(x["accents"]) <= demonstrated_accents for x in test)
    else:
        role_sets = [set() for _ in range(3)]
        for x in train:
            for i, color in enumerate((x["background"], x["frame"], x["accents"][0])):
                role_sets[i].add(color)
        assert any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3))
    if any(len(x["accents"]) == 2 for x in test):
        assert any(len(x["accents"]) == 2 for x in train)
    if any(x["chain"] for x in test):
        assert any(x["chain"] for x in train)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    parameters["observed_palettes"] = []
    total = train_count + test_count
    pair_seeds = rng.integers(0, 2**32, size=total, dtype=np.uint32)
    puzzle = {"train": [], "test": []}
    used = set()
    train_shapes = set()
    train_has_chain = False
    for global_index in range(total):
        section = "train" if global_index < train_count else "test"
        local_index = global_index if section == "train" else global_index - train_count
        regime = "source_train" if section == "train" else "target_test"
        pair_rng = np.random.default_rng(int(pair_seeds[global_index]))
        chain_demo = parameters["chain"] and ((section == "train" and local_index == 1) or
                                               (section == "test" and local_index == 0 and pair_rng.random() < 0.65))
        for attempt in range(16):
            inp, out = _generate_pair(pair_rng, regime, parameters, global_index, chain_demo)
            key = _grid_key(inp)
            rows, columns = np.where(inp == int(np.unique(out[inp != out])[0]))
            shape = (int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1))
            info = _inspect_pair(inp, out)
            if (key not in used and key not in _SOURCE_INPUT_KEYS and
                    (section != "train" or shape not in train_shapes) and
                    (section != "test" or train_has_chain or not info["chain"])):
                used.add(key)
                if section == "train":
                    train_shapes.add(shape)
                    train_has_chain |= info["chain"]
                    parameters["observed_palettes"].append(
                        (info["background"], info["frame"], *info["accents"])
                    )
                puzzle[section].append({"input": inp, "output": out})
                break
        else:
            raise AssertionError("unable to make a distinct frame puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and
               all(np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
                   for a, b in zip(first[section], second[section]) for name in ("input", "output"))
               for section in ("train", "test"))


def _expect_value_error(seed: int, num_train: int | None, num_test: int | None) -> None:
    try:
        generate_puzzle(seed, num_train, num_test)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert _puzzles_equal(generate_puzzle(91, 6, 3), generate_puzzle(91, 6, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, a, b in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                       (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        p = generate_puzzle(seed, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
        validate_generated_puzzle(p)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        _expect_value_error(300, a, b)
    shapes, canvases, margins, layouts, palettes = set(), set(), set(), set(), set()
    train_parities, test_parities, train_square, test_square = set(), set(), set(), set()
    counts, color_counts, chains = set(), set(), 0
    single, row_pair, column_pair, near, adjacent, zero_margin, dense = 0, 0, 0, 0, 0, 0, 0
    local_palette_puzzles, constant_palette_puzzles, two_accent_puzzles = 0, 0, 0
    background_changes, frame_changes, accent_changes, role_swaps = 0, 0, 0, 0
    same_regime_tests, larger_tests, thinner_tests, repositioned_tests = 0, 0, 0, 0
    combined_pairs, small, large, thin = 0, 0, 0, 0
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((section, tuple(_grid_key(pair["input"]) for pair in puzzle[section])) for section in ("train", "test")))
        train_info = [_inspect_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_inspect_pair(p["input"], p["output"]) for p in puzzle["test"]]
        structural_roles = {(x["background"], x["frame"]) for x in train_info}
        varied = len(structural_roles) > 1
        local_palette_puzzles += varied
        constant_palette_puzzles += not varied
        two_accent_puzzles += any(len(x["accents"]) == 2 for x in train_info)
        background_changes += len({x["background"] for x in train_info}) > 1
        frame_changes += len({x["frame"] for x in train_info}) > 1
        accent_changes += len({x["accents"] for x in train_info}) > 1
        role_swaps += any(a[0] == b[1] for a in structural_roles for b in structural_roles)
        train_regimes = {(x["frame_height"] % 2, x["frame_width"] % 2,
                          x["frame_height"] == x["frame_width"]) for x in train_info}
        same_regime_tests += any((x["frame_height"] % 2, x["frame_width"] % 2,
                                  x["frame_height"] == x["frame_width"]) in train_regimes for x in test_info)
        larger_tests += any(max(x["frame_height"], x["frame_width"]) >
                            max(max(y["frame_height"], y["frame_width"]) for y in train_info) for x in test_info)
        thinner_tests += any(max(x["frame_height"], x["frame_width"]) / min(x["frame_height"], x["frame_width"]) >
                             max(max(y["frame_height"], y["frame_width"]) / min(y["frame_height"], y["frame_width"]) for y in train_info) for x in test_info)
        repositioned_tests += any(x["left_margin"] not in {y["left_margin"] for y in train_info} for x in test_info)
        assert len({(x["frame_height"], x["frame_width"]) for x in train_info}) >= 2
        for section, pair_info in (("train", train_info), ("test", test_info)):
            for pair, info in zip(puzzle[section], pair_info):
                inp, out = pair["input"], pair["output"]
                _assert_valid_grid(inp)
                _assert_valid_grid(out)
                _validate_regime(info, "source_train" if section == "train" else "target_test")
                assert _grid_key(inp) not in _SOURCE_INPUT_KEYS
                h, w = info["frame_height"], info["frame_width"]
                # Independent cellwise assertion from the input geometry.
                frame_color = info["frame"]
                bg = info["background"]
                top, left = info["top_margin"], info["left_margin"]
                expected = inp.copy()
                for r in range(top + 1, top + h - 1):
                    for c in range(left + 1, left + w - 1):
                        if inp[r, c] == bg:
                            expected[r, c] = frame_color
                assert np.array_equal(expected, out)
                palettes.add((bg, frame_color, info["accents"]))
                shapes.add((h, w))
                canvases.add(inp.shape)
                margins.add((info["top_margin"], info["bottom_margin"], info["left_margin"], info["right_margin"]))
                layouts.add((h, w, info["accent_positions"]))
                counts.add(info["accent_count"])
                color_counts.add(len(info["accents"]))
                chains += info["chain"]
                single += info["accent_count"] == 1
                row_pair += info["accent_count"] == 2 and len({r for r, _ in info["accent_positions"]}) == 1
                column_pair += info["accent_count"] == 2 and len({c for _, c in info["accent_positions"]}) == 1
                near += info["near"]
                adjacent += info["adjacent"]
                zero_margin += any(info[k] == 0 for k in ("top_margin", "bottom_margin", "left_margin", "right_margin"))
                dense += info["accent_count"] / ((h - 2) * (w - 2)) >= 0.2
                combined_pairs += varied and (min(h, w) <= 4 or max(h, w) >= 16 or max(h, w) / min(h, w) >= 2.5) and (info["adjacent"] or info["near"] or info["chain"])
                small += min(h, w) <= 4
                large += max(h, w) >= 16
                thin += max(h, w) / min(h, w) >= 2.5
                if section == "train":
                    train_parities.add((h % 2, w % 2))
                    train_square.add(h == w)
                else:
                    test_parities.add((h % 2, w % 2))
                    test_square.add(h == w)
    assert len(signatures) >= 40
    assert len(shapes) >= 70 and len(canvases) >= 60 and len(margins) >= 60
    assert len(layouts) >= 100 and len(palettes) >= 20
    assert len(train_parities) == len(test_parities) == 4
    assert train_square == test_square == {True, False}
    assert len(counts) >= 8 and color_counts == {1, 2}
    assert min(single, row_pair, column_pair, near, adjacent, zero_margin, chains) > 0
    assert min(local_palette_puzzles, constant_palette_puzzles, two_accent_puzzles) > 0
    assert min(background_changes, frame_changes, accent_changes, role_swaps) > 0
    assert min(same_regime_tests, larger_tests, thinner_tests, repositioned_tests) > 0
    assert dense > 0 and combined_pairs > 0
    assert min(small, large, thin) > 0
