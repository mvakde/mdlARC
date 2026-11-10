import numpy as np


_AUTO_COUNT_PAIRS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 4),
    (7, 1),
    (8, 2),
    (9, 3),
    (10, 4),
)

_TEST_KINDS = ("narrow", "tall", "short", "wide")


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        (int(grid.shape[0]), int(grid.shape[1])),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _shear(input_grid: np.ndarray, background: int) -> np.ndarray:
    height, width = input_grid.shape
    output = np.full(
        (height, height + width - 1),
        background,
        dtype=input_grid.dtype,
    )
    for row in range(height):
        start = height - 1 - row
        output[row, start:start + width] = input_grid[row]
    return output


def _digit_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_INPUT_ROWS = (
    (
        "979777555",
        "479797755",
        "447797975",
        "444777979",
    ),
    (
        "488887",
        "148878",
        "414788",
        "641488",
        "664148",
        "666414",
    ),
    (
        "1666",
        "1666",
        "1666",
        "1888",
        "1555",
        "1555",
        "1555",
    ),
    (
        "1",
        "9",
        "5",
        "4",
    ),
    (
        "11",
        "11",
        "68",
        "68",
        "68",
        "68",
        "44",
        "44",
        "55",
        "55",
    ),
    (
        "5498",
        "8529",
    ),
)

_SOURCE_INPUTS = tuple(_digit_grid(rows) for rows in _SOURCE_INPUT_ROWS)

_SOURCE_PAIR_SIGNATURES = frozenset(
    (
        _grid_signature(input_grid),
        _grid_signature(_shear(input_grid, 0)),
    )
    for input_grid in _SOURCE_INPUTS
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")

    normalized = int(seed)
    if normalized < 0:
        raise ValueError("seed must be a nonnegative integer")
    return normalized


def _normalize_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count("num_train", num_train)
    test_count = _normalize_count("num_test", num_test)

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is None and test_count is None:
        return _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]

    if train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))

    if test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    background = int(rng.integers(0, 10))
    shared = bool(rng.random() < 0.20)
    colors = np.array([c for c in range(10) if c != background], dtype=np.int8)
    pool_size = int(rng.integers(3, 8))
    train_color_pool = rng.choice(colors, size=pool_size, replace=False).astype(np.int8)
    if rng.random() < 0.22:
        train_color_pool[int(rng.integers(0, pool_size))] = background
    shared_palette = None
    if shared:
        shared_palette = rng.choice(colors, size=2, replace=False).astype(np.int8)
    return {"background": background, "shared_palette": shared_palette,
            "train_color_pool": train_color_pool}


def _shape(rng: np.random.Generator, regime: str, anchor: bool = False) -> tuple[int, int]:
    if anchor:
        height = int(rng.integers(3, 13))
        width = int(rng.integers(2, min(16, 31 - height) + 1))
        return height, width

    kinds = ("small", "square", "tall", "wide", "boundary", "general")
    weights = (0.22, 0.17, 0.16, 0.16, 0.11, 0.18) if regime == "source_train" else (0.16, 0.13, 0.21, 0.19, 0.16, 0.15)
    kind = str(rng.choice(kinds, p=weights))
    if kind == "small":
        height = int(rng.integers(2, 9))
        width = int(rng.integers(1, min(11, 31 - height) + 1))
    elif kind == "square":
        height = int(rng.integers(4, 16))
        width = int(rng.integers(max(1, height - 3), min(height + 3, 31 - height) + 1))
    elif kind == "tall":
        height = int(rng.integers(8, 30))
        width = int(rng.integers(1, min(6, 31 - height) + 1))
    elif kind == "wide":
        height = int(rng.integers(1, 5))
        width = int(rng.integers(6, 32 - height))
    elif kind == "boundary":
        options = ((30, 1), (29, 2), (2, 29), (1, 30), (15, 16), (16, 15), (3, 28), (28, 3))
        height, width = options[int(rng.integers(0, len(options)))]
    else:
        height = int(rng.integers(1, 31))
        width = int(rng.integers(1 if height > 1 else 2, 32 - height))
    return height, width


def _palette(rng: np.random.Generator, background: int, height: int, width: int,
             shared_palette: np.ndarray | None, train_colors: set[int] | None,
             train_pool: np.ndarray, anchor: bool) -> tuple[np.ndarray, set[int]]:
    if shared_palette is not None:
        return shared_palette, set()
    maximum = min(8, height * width)
    minimum = 2 if anchor else 1
    size = int(rng.integers(minimum, maximum + 1))
    all_colors = set(range(10))
    if train_colors is None:
        size = min(size, len(train_pool))
        return rng.choice(train_pool, size=size, replace=False).astype(np.int8), set()

    familiar = sorted(train_colors)
    unseen = sorted(all_colors - train_colors)
    mode = int(rng.choice(4, p=(0.35, 0.28, 0.22, 0.15)))
    if mode == 0 and familiar:
        pool = familiar
    elif mode == 1 and unseen:
        # A single novel color can occur alone or beside familiar colors.
        chosen = [int(rng.choice(unseen))]
        pool = chosen + familiar
    elif mode == 2 and len(unseen) >= 2:
        chosen = [int(c) for c in rng.choice(unseen, size=2, replace=False)]
        pool = chosen + familiar + [c for c in unseen if c not in chosen]
    else:
        pool = sorted(all_colors)
    size = min(size, len(pool))
    if mode == 1 and unseen:
        required = [chosen[0]]
    elif mode == 2 and len(unseen) >= 2:
        required = chosen[:2]
    else:
        required = []
    remaining = [c for c in pool if c not in required]
    take = max(0, size - len(required))
    extras = rng.choice(np.array(remaining, dtype=np.int8), size=take, replace=False) if take else []
    return np.array(required + [int(c) for c in extras], dtype=np.int8), set(required)


def _payload(rng: np.random.Generator, height: int, width: int,
             palette: np.ndarray, anchor: bool = False) -> np.ndarray:
    count = len(palette)
    rows, cols = np.indices((height, width))
    style = int(rng.integers(0, 8)) if not anchor else int(rng.choice((1, 4, 5, 7)))
    if style == 0:  # uniform rectangles
        indices = np.full((height, width), int(rng.integers(0, count)), dtype=np.int64)
    elif style == 1:  # uneven color frequencies
        weights = rng.dirichlet(np.full(count, float(rng.uniform(0.35, 2.0))))
        indices = rng.choice(count, size=(height, width), p=weights)
    elif style == 2:  # row bands, with independent run lengths
        row_values = []
        while len(row_values) < height:
            run = int(rng.integers(1, min(6, height - len(row_values)) + 1))
            value = int(rng.integers(0, count))
            row_values.extend([value] * run)
        indices = np.broadcast_to(np.array(row_values[:height])[:, None], (height, width)).copy()
    elif style == 3:  # column bands
        col_values = []
        while len(col_values) < width:
            run = int(rng.integers(1, min(6, width - len(col_values)) + 1))
            col_values.extend([int(rng.integers(0, count))] * run)
        indices = np.broadcast_to(np.array(col_values[:width])[None, :], (height, width)).copy()
    elif style == 4:
        indices = (rows + cols + int(rng.integers(0, count))) % count
    elif style == 5:
        band = int(rng.integers(1, 4))
        indices = ((rows // band) + (cols // band) + int(rng.integers(0, count))) % count
    elif style == 6:  # a clean patch on a field
        indices = np.full((height, width), int(rng.integers(0, count)), dtype=np.int64)
        r0 = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width))
        r1 = int(rng.integers(r0 + 1, height + 1))
        c1 = int(rng.integers(c0 + 1, width + 1))
        indices[r0:r1, c0:c1] = int(rng.integers(0, count))
    else:  # identical rows in possibly unequal runs
        templates = rng.integers(0, count, size=(max(1, min(4, height)), width))
        row_values = []
        while len(row_values) < height:
            run = int(rng.integers(1, min(5, height - len(row_values)) + 1))
            row_values.extend([int(rng.integers(0, len(templates)))] * run)
        indices = templates[np.array(row_values[:height])]

    grid = palette[np.asarray(indices, dtype=np.int64)].astype(np.int8, copy=True)
    if anchor and len(np.unique(grid)) < 2:
        # Keep the first demonstration informative without imposing quotas elsewhere.
        grid[height - 1, width - 1] = palette[1] if grid[0, 0] == palette[0] else palette[0]
    if not anchor and count > 1 and style != 1 and rng.random() < 0.24:
        changes = int(rng.integers(1, max(2, height * width // 9 + 1)))
        positions = rng.choice(height * width, size=min(changes, height * width), replace=False)
        grid.flat[positions] = rng.choice(palette, size=len(positions))
    return grid


def _make_pair(rng: np.random.Generator, shape: tuple[int, int], palette: np.ndarray,
               background: int, anchor: bool, seen: set[tuple],
               require_palette: bool, required_colors: set[int]) -> dict:
    height, width = shape
    for _ in range(40):
        inp = _payload(rng, height, width, palette, anchor)
        if require_palette and set(int(c) for c in np.unique(inp)) != set(int(c) for c in palette):
            continue
        if required_colors and not required_colors.issubset(set(int(c) for c in np.unique(inp))):
            continue
        out = _shear(inp, background)
        key = _grid_signature(inp)
        if key not in seen and (key, _grid_signature(out)) not in _SOURCE_PAIR_SIGNATURES:
            seen.add(key)
            return {"input": inp, "output": out}
    # A new legal shape guarantees a unique fallback even for tiny uniform grids.
    for fallback_width in range(1, 32 - height):
        candidate = (height, fallback_width)
        if candidate == shape:
            continue
        inp = _payload(rng, *candidate, palette, anchor)
        if require_palette and set(int(c) for c in np.unique(inp)) != set(int(c) for c in palette):
            continue
        if required_colors and not required_colors.issubset(set(int(c) for c in np.unique(inp))):
            continue
        out = _shear(inp, background)
        key = _grid_signature(inp)
        if key not in seen and (key, _grid_signature(out)) not in _SOURCE_PAIR_SIGNATURES:
            seen.add(key)
            return {"input": inp, "output": out}
    raise RuntimeError("unable to construct a unique pair")


def _inferred_background(train: list[dict]) -> int:
    evidence = [int(pair["output"][0, 0]) for pair in train if pair["input"].shape[0] > 1]
    assert evidence and len(set(evidence)) == 1
    return evidence[0]


def _assert_pair(inp: np.ndarray, out: np.ndarray, background: int) -> None:
    height, width = inp.shape
    assert out.shape == (height, height + width - 1)
    for r in range(height):
        left = height - 1 - r
        right = left + width
        assert np.array_equal(out[r, left:right], inp[r])
        assert np.all(out[r, :left] == background)
        assert np.all(out[r, right:] == background)
    if height > 1:
        assert not np.array_equal(inp, out)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    background = _inferred_background(puzzle["train"])
    assert 0 <= background <= 9
    train_shapes = set()
    informative = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape[0] + inp.shape[1] - 1 <= 30
            _assert_pair(inp, out, background)
            key = _grid_signature(inp)
            pair_key = (key, _grid_signature(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_SIGNATURES
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_shapes.add(inp.shape)
                if inp.shape[0] >= 3 and inp.shape[1] >= 2 and len(np.unique(inp)) >= 2:
                    informative = True
    assert informative and len(train_shapes) >= 2


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    background = parameters["background"]
    shared = parameters["shared_palette"]
    train_pool = parameters["train_color_pool"]
    seen: set[tuple] = set()
    train = []
    for index in range(train_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        shape = _shape(pair_rng, "source_train", anchor=index == 0)
        if index == 1:
            for _ in range(8):
                if shape != train[0]["input"].shape:
                    break
                shape = _shape(pair_rng, "source_train")
        palette, required = _palette(pair_rng, background, *shape, shared, None, train_pool, index == 0)
        train.append(_make_pair(pair_rng, shape, palette, background, index == 0,
                                seen, shared is not None, required))
    train_colors = {int(c) for pair in train for c in np.unique(pair["input"])}
    test = []
    for _ in range(test_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        shape = _shape(pair_rng, "target_test")
        palette, required = _palette(pair_rng, background, *shape, shared, train_colors,
                                     train_pool, False)
        test.append(_make_pair(pair_rng, shape, palette, background, False, seen,
                               shared is not None, required))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(train: list[dict], inp: np.ndarray) -> np.ndarray:
    # Infer the only puzzle-level convention from visible training padding.
    background = _inferred_background(train)
    height, width = inp.shape
    rows = []
    for r in range(height):
        rows.append([background] * (height - 1 - r) + [int(c) for c in inp[r]] + [background] * r)
    return np.array(rows, dtype=np.int8)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) >= 40
    assert _puzzle_signature(generate_puzzle(17)) == _puzzle_signature(generate_puzzle(17))
    backgrounds, shapes, palette_sizes, signatures = set(), set(), set(), set()
    familiar_tests = novel_tests = multiple_novel_tests = 0
    varied_train_palettes = 0
    shared_train_palettes = 0
    clean_repeated = unequal_runs = row_bands = column_bands = 0
    clean_diagonals = clean_patches = 0
    uniform = rich = payload_background = zero_content = 0
    blank_rows = blank_columns = boundary = single_rows = single_columns = 0
    different_novel_sets = overlapping_heights = taller_tests = shorter_tests = 0
    combined_tests = 0
    observed_shapes = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background = _inferred_background(puzzle["train"])
        backgrounds.add(background)
        train_colors = {int(c) for pair in puzzle["train"] for c in np.unique(pair["input"])}
        train_palettes = {tuple(int(c) for c in np.unique(pair["input"])) for pair in puzzle["train"]}
        train_heights = [pair["input"].shape[0] for pair in puzzle["train"]]
        varied_train_palettes += len(train_palettes) > 1
        shared_train_palettes += len(train_palettes) == 1
        test_novel_sets = []
        combined_here = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _assert_pair(inp, out, background)
                signatures.add(_grid_signature(inp))
                shapes.add((split, inp.shape))
                observed_shapes.add(inp.shape)
                colors = set(int(c) for c in np.unique(inp))
                palette_sizes.add(len(colors))
                payload_background += np.any(inp == background)
                zero_content += background != 0 and 0 in colors
                boundary += inp.shape[0] + inp.shape[1] - 1 == 30
                single_rows += inp.shape[0] == 1
                single_columns += inp.shape[1] == 1
                uniform += len(colors) == 1
                rich += len(colors) >= 5
                blank_rows += any(np.all(row == background) for row in inp)
                blank_columns += any(np.all(inp[:, c] == background) for c in range(inp.shape[1]))
                row_patterns = [tuple(row) for row in inp]
                clean_repeated += len(set(row_patterns)) < inp.shape[0]
                runs = []
                for row_pattern in row_patterns:
                    if not runs or runs[-1][0] != row_pattern:
                        runs.append([row_pattern, 1])
                    else:
                        runs[-1][1] += 1
                unequal_runs += max(v for _, v in runs) >= 2 and len({v for _, v in runs}) >= 2
                row_bands += len(colors) > 1 and all(len(set(row)) == 1 for row in inp)
                column_bands += len(colors) > 1 and all(len(set(inp[:, c])) == 1 for c in range(inp.shape[1]))
                if len(colors) >= 2 and min(inp.shape) >= 2:
                    ordered = sorted(colors)
                    mapped = np.empty(inp.shape, dtype=np.int8)
                    for color_index, color in enumerate(ordered):
                        mapped[inp == color] = color_index
                    diagonal = (np.indices(inp.shape).sum(axis=0) + int(mapped[0, 0])) % len(colors)
                    clean_diagonals += np.array_equal(mapped, diagonal)
                if len(colors) == 2 and min(inp.shape) >= 4:
                    values, counts = np.unique(inp, return_counts=True)
                    minority = values[int(np.argmin(counts))]
                    rr, cc = np.where(inp == minority)
                    clean_patches += (len(rr) >= 4 and len(rr) < inp.size / 3 and
                                      len(rr) == (int(rr.max()) - int(rr.min()) + 1) *
                                      (int(cc.max()) - int(cc.min()) + 1))
                if split == "test":
                    unseen = colors - train_colors
                    test_novel_sets.append(frozenset(unseen))
                    familiar_tests += len(unseen) == 0
                    novel_tests += len(unseen) > 0
                    multiple_novel_tests += len(unseen) > 1
                    overlapping_heights += min(train_heights) <= inp.shape[0] <= max(train_heights)
                    taller_tests += inp.shape[0] > max(train_heights)
                    shorter_tests += inp.shape[0] < min(train_heights)
                    combined_here |= bool(unseen) and (inp.shape[0] == 1 or inp.shape[1] == 1 or
                                                         inp.shape[0] + inp.shape[1] - 1 == 30 or
                                                         background in colors)
                    copied_train = [{"input": pair["input"].copy(), "output": pair["output"].copy()}
                                    for pair in puzzle["train"]]
                    assert np.array_equal(_predict_from_training(copied_train, inp.copy()), out)
        different_novel_sets += len({v for v in test_novel_sets if v}) >= 2
        combined_tests += combined_here
    assert len(backgrounds) >= 8
    assert len(shapes) >= 65 and len(signatures) >= 250
    assert len(palette_sizes) >= 7
    assert varied_train_palettes >= 20 and shared_train_palettes >= 1
    assert familiar_tests >= 12 and novel_tests >= 12 and multiple_novel_tests >= 8
    assert different_novel_sets >= 3 and combined_tests >= 5
    assert clean_repeated >= 10 and unequal_runs >= 8
    assert row_bands >= 4 and column_bands >= 4 and uniform >= 5 and rich >= 5
    assert clean_diagonals >= 3 and clean_patches >= 2
    assert payload_background >= 5 and zero_content >= 5
    assert blank_rows >= 2 and blank_columns >= 2
    assert boundary >= 6 and single_rows >= 2 and single_columns >= 2
    assert overlapping_heights >= 10 and taller_tests >= 2 and shorter_tests >= 2
    observed_shapes.update(pair["input"].shape for split in ("train", "test")
                           for pair in generate_puzzle(100)[split])
    assert {(2, 2), (6, 3), (2, 29), (29, 2), (30, 1)} <= observed_shapes
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 2)):
        puzzle = generate_puzzle(1000 + train_count * 10 + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train_count or 0) + (test_count or 0), train_count, test_count)
        assert (train_count is None or len(puzzle["train"]) == train_count)
        assert (test_count is None or len(puzzle["test"]) == test_count)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
