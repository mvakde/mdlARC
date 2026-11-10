import numpy as np


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


def _grid_from_text(text: str) -> np.ndarray:
    rows = text.split("/")
    return np.array([[int(char) for char in row] for row in rows], dtype=np.int8)


_SOURCE_TEXT_PAIRS = (
    (
        "00000000/14144141/00000000/00000000/23000000/00000000/"
        "82000000/00000000/65000000/00000000/00000000/00000000",
        "00000000/14144141/00000000/00000000/23233232/00000000/"
        "82822828/00000000/65655656/00000000/00000000/00000000",
    ),
    (
        "00000000/22121211/00000000/33100000/00000000/00000000/"
        "00000000/88200000/00000000/00000000",
        "00000000/22121211/00000000/33131311/00000000/00000000/"
        "00000000/88282822/00000000/00000000",
    ),
    (
        "00000000/33444343/00000000/00000000/88200000/00000000",
        "00000000/33444343/00000000/00000000/88222828/00000000",
    ),
    (
        "00000000/33233233/00000000/00000000/88400000/00000000/"
        "11600000/00000000/00000000/00000000",
        "00000000/33233233/00000000/00000000/88488488/00000000/"
        "11611611/00000000/00000000/00000000",
    ),
    (
        "0000000000/2212121121/0000000000/8830000000/0000000000/"
        "0000000000/1140000000/0000000000/0000000000/6680000000/"
        "0000000000/1160000000/0000000000/0000000000",
        "0000000000/2212121121/0000000000/8838383383/0000000000/"
        "0000000000/1141414414/0000000000/0000000000/6686868868/"
        "0000000000/1161616616/0000000000/0000000000",
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(_grid_from_text(input_text), _grid_from_text(output_text))
    for input_text, output_text in _SOURCE_TEXT_PAIRS
)


def _validate_count_argument(
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
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count_argument("num_train", num_train, 2, 10)
    num_test = _validate_count_argument("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        largest_test_count = min(4, num_train - 1)
        if largest_test_count < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, largest_test_count + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The direction of alignment is the sole convention shared by a puzzle.
    convention = str(rng.choice(
        ["horizontal_left", "horizontal_right", "vertical_top", "vertical_bottom"],
        p=[0.67, 0.13, 0.12, 0.08],
    ))
    base_background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    return {"convention": convention, "base_background": base_background}


def _short_period(roles: np.ndarray) -> bool:
    width = len(roles)
    return any(
        all(roles[i] == roles[i % period] for i in range(width))
        for period in range(1, width // 2 + 1)
    )


def _sample_roles(rng: np.random.Generator, width: int, witness: bool) -> np.ndarray:
    mode = "random" if witness else str(rng.choice(
        ["random", "periodic", "run"], p=[0.58, 0.25, 0.17]
    ))
    if mode == "periodic":
        initial_run = int(rng.integers(1, min(5, width // 2)))
        minimum_period = initial_run + 1
        period = int(rng.integers(minimum_period, min(7, width // 2) + 1))
        motif = rng.integers(0, 2, size=period, dtype=np.int8)
        motif[:initial_run] = 0
        motif[initial_run] = 1
        return np.resize(motif, width).astype(np.int8)

    maximum_run = min(3 if witness else 5, width - 2)
    initial_run = int(rng.integers(1, maximum_run + 1))
    if mode == "run":
        roles = np.ones(width, dtype=np.int8)
        roles[:initial_run] = 0
        return roles

    for _ in range(64):
        roles = rng.integers(0, 2, size=width, dtype=np.int8)
        roles[:initial_run] = 0
        roles[initial_run] = 1
        if not witness or (
            not _short_period(roles)
            and not np.array_equal(roles, np.resize(roles[:initial_run + 1], width))
        ):
            return roles

    # This bounded fallback keeps the training witness nonperiodic.
    roles = np.ones(width, dtype=np.int8)
    roles[:initial_run] = 0
    roles[-1] = 0
    return roles


def _sample_palettes(
    rng: np.random.Generator, background: int, target_count: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    colors = np.array([color for color in range(10) if color != background], dtype=np.int8)
    reference = rng.choice(colors, size=2, replace=False).astype(np.int8)
    mode = str(rng.choice(["random", "repeat", "reverse", "share", "disjoint"]))
    if mode == "disjoint" and 2 * target_count <= len(colors):
        drawn = rng.choice(colors, size=2 * target_count, replace=False)
        targets = [drawn[2 * i:2 * i + 2].astype(np.int8) for i in range(target_count)]
    else:
        targets = [rng.choice(colors, size=2, replace=False).astype(np.int8)
                   for _ in range(target_count)]
        if target_count >= 2:
            if mode == "repeat":
                targets[1] = targets[0].copy()
            elif mode == "reverse":
                targets[1] = targets[0][::-1].copy()
            elif mode == "share":
                shared = int(rng.integers(0, 2))
                other_choices = colors[(colors != targets[0][0]) & (colors != targets[0][1])]
                other = int(rng.choice(other_choices))
                targets[1] = np.array([targets[0][shared], other], dtype=np.int8)
                if rng.random() < 0.5:
                    targets[1] = targets[1][::-1].copy()
    return reference, targets


def _sample_prefix(
    rng: np.random.Generator, minimum: int, width: int
) -> int:
    if minimum == width - 1:
        return minimum
    roll = rng.random()
    if roll < 0.48:
        return minimum
    if roll < 0.86:
        return int(rng.integers(minimum + 1, min(width, minimum + 6)))
    return int(rng.integers(max(minimum + 1, width - 3), width))


def _sample_layout(
    rng: np.random.Generator, target_count: int
) -> tuple[int, int, list[int]]:
    lane_count = target_count + 1
    top = int(rng.integers(0, 4))
    bottom = int(rng.integers(0, 4))
    if top + bottom == 0:
        bottom = 1
    gaps = rng.integers(1, 4, size=target_count)
    lanes = [top]
    for gap in gaps:
        lanes.append(lanes[-1] + int(gap) + 1)
    height = lanes[-1] + bottom + 1
    assert height <= 30 and len(lanes) == lane_count

    if target_count == 1:
        reference_index = 0 if rng.random() < 0.65 else 1
    else:
        placement = rng.random()
        if placement < 0.43:
            reference_index = 0
        elif placement < 0.59:
            reference_index = target_count
        else:
            reference_index = int(rng.integers(1, target_count))
    reference_row = lanes[reference_index]
    target_rows = [row for i, row in enumerate(lanes) if i != reference_index]
    return height, reference_row, target_rows


def _display_grid(grid: np.ndarray, convention: str) -> np.ndarray:
    if convention == "horizontal_left":
        return grid.copy()
    if convention == "horizontal_right":
        return grid[:, ::-1].copy()
    if convention == "vertical_top":
        return grid.T.copy()
    if convention == "vertical_bottom":
        return grid.T[::-1, :].copy()
    raise ValueError("unknown alignment convention")


def _canonical_grid(grid: np.ndarray, convention: str) -> np.ndarray:
    if convention == "horizontal_left":
        return grid.copy()
    if convention == "horizontal_right":
        return grid[:, ::-1].copy()
    if convention == "vertical_top":
        return grid.T.copy()
    if convention == "vertical_bottom":
        return grid[::-1, :].T.copy()
    raise ValueError("unknown alignment convention")


def generate_pair(
    rng: np.random.Generator, regime: str, puzzle_parameters: dict,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train":
        width = int(rng.integers(9, 18)) if pair_index == 0 else int(rng.integers(7, 20))
        target_count = int(rng.integers(2, 4)) if pair_index == 0 else int(rng.integers(1, 5))
    elif regime == "target_test":
        width = int(rng.integers(14, 23)) if rng.random() < 0.43 else int(rng.integers(7, 19))
        target_count = int(rng.integers(2, 6)) if rng.random() < 0.68 else int(rng.integers(1, 5))
    else:
        raise ValueError("unknown pair regime")

    background = (puzzle_parameters["base_background"] if rng.random() < 0.67
                  else int(rng.integers(0, 10)))
    roles = _sample_roles(rng, width, witness=regime == "source_train" and pair_index == 0)
    height, reference_row, target_rows = _sample_layout(rng, target_count)
    reference_palette, target_palettes = _sample_palettes(rng, background, target_count)
    first_other = int(np.flatnonzero(roles != roles[0])[0])
    minimum = first_other + 1
    prefixes = [_sample_prefix(rng, minimum, width) for _ in target_rows]
    if regime == "source_train" and pair_index == 0:
        prefixes[0] = minimum
        prefixes[1] = int(rng.integers(minimum + 1, width))

    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    reference = reference_palette[roles]
    inp[reference_row] = reference
    out[reference_row] = reference
    for row, palette, prefix in zip(target_rows, target_palettes, prefixes):
        completed = palette[roles]
        inp[row, :prefix] = completed[:prefix]
        out[row] = completed
    convention = puzzle_parameters["convention"]
    return _display_grid(inp, convention), _display_grid(out, convention)


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    maximum = int(np.max(counts))
    assert int(np.count_nonzero(counts == maximum)) == 1
    return int(colors[int(np.argmax(counts))])


def _complete_canonical_input(inp: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = inp.shape
    assert 7 <= width <= 22 and 4 <= height <= 30
    background = _background(inp)
    active = [row for row in range(height) if np.any(inp[row] != background)]
    assert 2 <= len(active) <= 6
    assert all(b - a >= 2 for a, b in zip(active, active[1:]))
    full = [row for row in active if np.all(inp[row] != background)]
    assert len(full) == 1
    reference_row = full[0]
    reference = inp[reference_row]
    assert len(np.unique(reference)) == 2
    first_other = int(np.flatnonzero(reference != reference[0])[0])
    assert first_other <= width - 2
    first = int(reference[0])
    second = int(reference[first_other])
    roles = (reference == second).astype(np.int8)
    out = inp.copy()
    target_rows = [row for row in active if row != reference_row]
    prefixes = []
    palettes = []
    for row in target_rows:
        colored = np.flatnonzero(inp[row] != background)
        prefix = len(colored)
        assert first_other + 1 <= prefix < width
        assert np.array_equal(colored, np.arange(prefix))
        palette = (int(inp[row, 0]), int(inp[row, first_other]))
        assert palette[0] != palette[1]
        completed = np.where(roles == 0, palette[0], palette[1]).astype(inp.dtype)
        assert np.array_equal(inp[row, :prefix], completed[:prefix])
        out[row] = completed
        prefixes.append(prefix)
        palettes.append(palette)
    info = {
        "height": height, "width": width, "background": background,
        "reference_row": reference_row, "target_rows": tuple(target_rows),
        "target_count": len(target_rows), "initial_run": first_other,
        "prefixes": tuple(prefixes), "palettes": tuple(palettes),
        "reference_palette": (first, second), "role_pattern": tuple(map(int, roles)),
        "short_period": _short_period(roles),
        "one_switch": int(np.count_nonzero(roles[1:] != roles[:-1])) == 1,
        "prefix_tile": np.array_equal(roles, np.resize(roles[:first_other + 1], width)),
    }
    return out, info


_CONVENTIONS = (
    "horizontal_left", "horizontal_right", "vertical_top", "vertical_bottom"
)


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    assert inp.shape == out.shape
    matches = []
    for convention in _CONVENTIONS:
        try:
            canonical = _canonical_grid(inp, convention)
            expected, info = _complete_canonical_input(canonical)
            assert np.array_equal(_display_grid(expected, convention), out)
            matches.append((convention, info))
        except (AssertionError, IndexError):
            pass
    assert len(matches) == 1
    convention, info = matches[0]
    assert not np.array_equal(inp, out)
    return {"convention": convention, **info}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(inp)
            pair_key = _pair_key(inp, out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_analyze_pair(inp, out))
    conventions = {info["convention"] for split in infos for info in infos[split]}
    assert len(conventions) == 1
    # A nonperiodic training example with a shortest key rules out merely tiling
    # the visible target prefix. Both short and longer keys are demonstrated.
    assert any(
        not info["short_period"] and not info["prefix_tile"]
        and info["initial_run"] + 1 in info["prefixes"]
        for info in infos["train"]
    )
    assert any(len(set(info["prefixes"])) >= 2 for info in infos["train"])
    assert all(7 <= info["width"] <= 19 for info in infos["train"])
    assert all(7 <= info["width"] <= 22 for info in infos["test"])
    assert all(1 <= info["target_count"] <= 4 for info in infos["train"])
    assert all(1 <= info["target_count"] <= 5 for info in infos["test"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    index = 0
    for split, regime, count in (
        ("train", "source_train", train_count),
        ("test", "target_test", test_count),
    ):
        for pair_index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[index]))
            index += 1
            for _ in range(40):
                inp, out = generate_pair(pair_rng, regime, parameters, pair_index)
                key = _grid_key(inp)
                if key not in seen and _pair_key(inp, out) not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            puzzle[split].append({"input": inp, "output": out})
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple(_pair_key(pair["input"], pair["output"]) for pair in puzzle[split])
        for split in ("train", "test")
    )


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check receives only demonstrations and the unsolved input.
    conventions = {_analyze_pair(pair["input"], pair["output"])["convention"]
                   for pair in train}
    assert len(conventions) == 1
    convention = next(iter(conventions))
    canonical = _canonical_grid(test_input, convention)
    completed, _ = _complete_canonical_input(canonical)
    return _display_grid(completed, convention)


def _expect_value_error(**arguments) -> None:
    try:
        generate_puzzle(123, **arguments)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {arguments}")


def self_test() -> None:
    train_counts, test_counts, conventions = set(), set(), set()
    widths, heights, backgrounds, target_counts, initial_runs = set(), set(), set(), set(), set()
    layouts, role_patterns, signatures, puzzle_keys = set(), set(), set(), set()
    periodic = simple_run = extended = mixed_prefix = mixed_side = 0
    repeated = reversed_palette = distinct = shared = reference_reuse = 0
    smaller_test = larger_test = varied_background = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test_infos = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        assert {x["convention"] for x in train_infos + test_infos} == {train_infos[0]["convention"]}
        conventions.add(train_infos[0]["convention"])
        varied_background += len({x["background"] for x in train_infos + test_infos}) > 1
        smaller_test += any(t["width"] <= s["width"] and t["height"] <= s["height"]
                            for t in test_infos for s in train_infos)
        larger_test += any(t["width"] > s["width"] or t["target_count"] > s["target_count"]
                           for t in test_infos for s in train_infos)
        for pair, info in zip(puzzle["test"], test_infos):
            predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        for info in train_infos + test_infos:
            widths.add(info["width"])
            heights.add(info["height"])
            backgrounds.add(info["background"])
            target_counts.add(info["target_count"])
            initial_runs.add(info["initial_run"])
            layouts.add((info["height"], info["reference_row"], info["target_rows"]))
            role_patterns.add(info["role_pattern"])
            signatures.add((info["convention"], info["height"], info["width"],
                            info["target_count"], info["reference_row"], info["initial_run"],
                            info["prefixes"], info["short_period"]))
            periodic += info["short_period"]
            simple_run += info["one_switch"]
            extended += any(p > info["initial_run"] + 1 for p in info["prefixes"])
            mixed_prefix += len(set(info["prefixes"])) > 1
            mixed_side += min(info["target_rows"]) < info["reference_row"] < max(info["target_rows"])
            palettes = info["palettes"]
            repeated += len(set(palettes)) < len(palettes)
            reversed_palette += any(a == b[::-1] for i, a in enumerate(palettes)
                                    for b in palettes[i + 1:])
            distinct += len({color for palette in palettes for color in palette}) == 2 * len(palettes)
            shared += len({color for palette in palettes for color in palette}) < 2 * len(palettes)
            reference_reuse += bool(set(info["reference_palette"]) &
                                    {color for palette in palettes for color in palette})
    assert len(puzzle_keys) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(conventions) >= 3 and "horizontal_left" in conventions
    assert len(widths) >= 12 and len(heights) >= 12
    assert len(backgrounds) >= 7 and target_counts == {1, 2, 3, 4, 5}
    assert len(initial_runs) >= 4 and len(layouts) >= 80 and len(role_patterns) >= 80
    assert len(signatures) >= 120
    assert min(periodic, simple_run, extended, mixed_prefix, mixed_side) > 0
    assert min(repeated, reversed_palette, distinct, shared, reference_reuse) > 0
    assert min(smaller_test, larger_test, varied_background) > 0
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 3)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(121, train, test))
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        _expect_value_error(**arguments)
