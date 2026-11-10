"""Generate centered-block-to-corners ARC puzzles."""

import numpy as np


_SOURCE_PAYLOADS = {(5, 6, 8, 3), (3, 4, 7, 6), (2, 3, 4, 9)}
_SIZES = tuple(range(4, 31, 2))
_SIZE_WEIGHTS = np.array([25, 14, 11, 9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 1], dtype=float)
_SIZE_WEIGHTS /= _SIZE_WEIGHTS.sum()
_COLORS = tuple(range(10))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _integer(value, name: str, low: int, high: int | None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        if high is None:
            raise ValueError(f"{name} must be at least {low}")
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        if seed % 5 == 0:
            return 2, 1
        return legal[seed % len(legal)]
    if num_train is None:
        options = list(range(max(2, num_test + 1), 11))
        num_train = options[seed % len(options)]
    if num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choice(rng: np.random.Generator, values) -> int:
    return int(rng.choice(values))


def _sample_sizes(rng: np.random.Generator, num_train: int, num_test: int):
    # The shared, seen-multiscale, and held-out-scale regimes all use even squares.
    regime = _choice(rng, (0, 1, 2))
    if regime == 0:
        size = _choice(rng, _SIZES) if rng.random() >= 0.55 else 4
        return [size] * num_train, [size] * num_test, regime

    count = int(rng.integers(2, min(5, num_train) + 1))
    vocabulary = [int(x) for x in rng.choice(_SIZES, size=count, replace=False, p=_SIZE_WEIGHTS)]
    train_sizes = vocabulary + [_choice(rng, vocabulary) for _ in range(num_train - count)]
    rng.shuffle(train_sizes)
    if regime == 1:
        test_sizes = [_choice(rng, vocabulary) for _ in range(num_test)]
    else:
        unseen = [size for size in _SIZES if size not in vocabulary]
        # Give the unseen scale substantial representation among test pairs.
        test_sizes = [_choice(rng, unseen)]
        test_sizes += [_choice(rng, unseen if rng.random() < 0.65 else vocabulary)
                       for _ in range(num_test - 1)]
        rng.shuffle(test_sizes)
    return train_sizes, test_sizes, regime


def _grids(size: int, background: int, payload: tuple[int, int, int, int]):
    inp = np.full((size, size), background, dtype=np.int8)
    out = np.full((size, size), background, dtype=np.int8)
    mid = size // 2
    for value, (r, c), (y, x) in zip(
        payload,
        ((mid - 1, mid - 1), (mid - 1, mid), (mid, mid - 1), (mid, mid)),
        ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)),
    ):
        inp[r, c] = value
        out[y, x] = value
    return {"input": inp, "output": out}


def _payload(rng: np.random.Generator, familiar: tuple[int, ...],
             novel: tuple[int, ...], background: int, repeated: bool):
    familiar = tuple(color for color in familiar if color != background)
    novel = tuple(color for color in novel if color != background)
    distinct_count = _choice(rng, (2, 3)) if repeated else 4
    possible_novel = [n for n in range(distinct_count + 1)
                      if n <= len(novel) and distinct_count - n <= len(familiar)]
    novel_count = _choice(rng, possible_novel)
    selected = [int(x) for x in rng.choice(novel, size=novel_count, replace=False)]
    selected += [int(x) for x in rng.choice(familiar, size=distinct_count - novel_count,
                                            replace=False)]
    if repeated:
        selected += [selected[_choice(rng, range(distinct_count))]
                     for _ in range(4 - distinct_count)]
    rng.shuffle(selected)
    return tuple(selected)


def _signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, None)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The geometric convention is constant. These choices set palette and scale
    # opportunities for the puzzle, never a hidden rule required by a solver.
    if num_train == 2 and num_test == 1 and (seed % 5 == 0 or rng.random() < 0.45):
        train_sizes, test_sizes = [4, 4], [4]
    else:
        train_sizes, test_sizes, _ = _sample_sizes(rng, num_train, num_test)
    maximum_palette = 6 if num_train == 2 else 8
    palette_size = int(rng.integers(4, maximum_palette + 1))
    train_palette = tuple(int(x) for x in rng.choice(_COLORS, size=palette_size,
                                                      replace=False))
    background_options = tuple(color for color in _COLORS if color not in train_palette)
    stable_background = bool(rng.integers(0, 2))
    if stable_background:
        base = 0 if 0 in background_options and rng.random() < 0.55 else _choice(rng, background_options)
        train_backgrounds = [base] * num_train
        test_backgrounds = [base] * num_test
    else:
        first_two = [int(x) for x in rng.choice(background_options, size=2,
                                                 replace=False)]
        train_backgrounds = first_two + [_choice(rng, background_options)
                                         for _ in range(num_train - 2)]
        test_backgrounds = [_choice(rng, background_options) for _ in range(num_test)]

    # Two examples share colors in different quadrants, ruling out a fixed
    # color-to-corner lookup. A third example exposes any remaining colors.
    first_values = tuple(int(x) for x in rng.choice(train_palette, size=4, replace=False))
    missing = [color for color in train_palette if color not in first_values]
    second_missing = missing[:2]
    second_values = second_missing + [int(x) for x in rng.choice(
        first_values, size=4 - len(second_missing), replace=False)]
    for _ in range(100):
        rng.shuffle(second_values)
        shared = set(first_values) & set(second_values)
        if any(first_values.index(value) != second_values.index(value) for value in shared):
            break
    planned = [first_values, tuple(second_values)]
    remaining = missing[2:]
    if remaining:
        third_values = remaining + [int(x) for x in rng.choice(
            [color for color in train_palette if color not in remaining],
            size=4 - len(remaining), replace=False)]
        rng.shuffle(third_values)
        planned.append(tuple(third_values))

    puzzle = {"train": [], "test": []}
    seen = set()
    for index, size in enumerate(train_sizes):
        background = train_backgrounds[index]
        if index < len(planned):
            candidate = planned[index]
        else:
            candidate = None
            if rng.random() < 0.50:
                prior = [pair for pair in puzzle["train"]
                         if pair["input"].shape[0] != size
                         and int(pair["input"][0, 0]) == background]
                if prior:
                    example = prior[int(rng.integers(len(prior)))]["input"]
                    m = example.shape[0] // 2
                    candidate = tuple(int(x) for x in example[m - 1:m + 1,
                                                               m - 1:m + 1].flat)
            if candidate is None:
                candidate = _payload(rng, train_palette, (), background,
                                     repeated=bool(rng.random() < 0.20))
        for attempt in range(300):
            if attempt:
                candidate = _payload(rng, train_palette, (), background,
                                     repeated=index >= 2 and bool(rng.random() < 0.20))
            pair = _grids(size, background, candidate)
            key = _signature(pair["input"])
            if key not in seen and not (size == 4 and background == 0
                                        and candidate in _SOURCE_PAYLOADS):
                break
        else:
            raise RuntimeError("could not sample a unique training pair")
        seen.add(key)
        puzzle["train"].append(pair)

    novel_palette = tuple(color for color in _COLORS if color not in train_palette)
    for size, background in zip(test_sizes, test_backgrounds):
        for _ in range(300):
            candidate = _payload(rng, train_palette, novel_palette, background,
                                 repeated=bool(rng.random() < 0.28))
            pair = _grids(size, background, candidate)
            key = _signature(pair["input"])
            if key not in seen and not (size == 4 and background == 0
                                        and candidate in _SOURCE_PAYLOADS):
                break
        else:
            raise RuntimeError("could not sample a unique test pair")
        seen.add(key)
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair: dict):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must contain input and output")
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "grids must be two-dimensional NumPy arrays")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                 "grid dimensions out of ARC bounds")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integral")
        _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors out of ARC range")
    _require(inp.shape == out.shape, "output dimensions must match input")
    height, width = inp.shape
    _require(height == width and height >= 4 and height % 2 == 0,
             "grids must be even squares of side at least four")
    background = int(inp[0, 0])
    mid = height // 2
    payload = tuple(int(x) for x in inp[mid - 1:mid + 1, mid - 1:mid + 1].flat)
    _require(all(value != background for value in payload),
             "all four center cells must differ from background")
    expected = _grids(height, background, payload)
    _require(np.array_equal(inp, expected["input"]),
             "input must contain only a centered filled 2x2 block")
    _require(np.array_equal(out, expected["output"]),
             "output must preserve values in matching outer corners")
    _require(not np.array_equal(inp, out), "input and output must differ")
    _require(not (height == 4 and background == 0 and payload in _SOURCE_PAYLOADS),
             "generated pair copies a source pair")
    return height, background, payload


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "pair counts are invalid")
    seen_inputs, seen_pairs = set(), set()
    train_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair)
            input_key = _signature(pair["input"])
            pair_key = input_key, _signature(pair["output"])
            _require(input_key not in seen_inputs, "duplicate input")
            _require(pair_key not in seen_pairs, "duplicate input/output pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_analyses.append(analysis)
    _require(all(len(set(payload)) == 4 for _, _, payload in train_analyses[:2]),
             "first two training examples must identify all four roles")
    first, second = train_analyses[0][2], train_analyses[1][2]
    shared = set(first) & set(second)
    _require(len(shared) >= 2 and any(first.index(value) != second.index(value)
                                      for value in shared),
             "training must move a shared color to a different quadrant")
    # Each training example explicitly demonstrates every quadrant mapping.


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    from collections import Counter
    counts = Counter()
    signatures = set()
    palette_sizes = set()
    for seed in range(96):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train = [_analyze_pair(p) for p in puzzle["train"]]
        test = [_analyze_pair(p) for p in puzzle["test"]]
        train_colors = set(x for _, _, payload in train for x in payload)
        palette_sizes.add(len(train_colors))
        train_sizes = {size for size, _, _ in train}
        all_pairs = train + test
        varying_background = len({bg for _, bg, _ in all_pairs}) > 1
        heldout_size = any(size not in train_sizes for size, _, _ in test)
        repeated_test = any(len(set(payload)) < 4 for _, _, payload in test)
        counts[("train_count", len(train))] += 1
        counts[("test_count", len(test))] += 1
        counts[("scale_regime", len(train_sizes) == 1,
                heldout_size)] += 1
        counts[("source_like", len(train) == 2 and len(test) == 1
                and all(size == 4 for size, _, _ in all_pairs))] += 1
        counts[("backgrounds", varying_background)] += 1
        counts[("zero_payload", any(0 in payload for _, _, payload in all_pairs))] += 1
        counts[("repeat_test", repeated_test)] += 1
        counts[("heldout_and_background", heldout_size and varying_background)] += 1
        counts[("repeat_and_background", repeated_test and varying_background)] += 1
        counts[("test_palette_variety", len(test) >= 2 and len({frozenset(payload)
                for _, _, payload in test}) >= 2)] += 1
        counts[("permuted_set", any(set(a) == set(b) and a != b
                for i, (_, _, a) in enumerate(train)
                for _, _, b in train[i + 1:]))] += 1
        counts[("compact_pair", any(size <= 8 for size, _, _ in all_pairs))] += 1
        counts[("large_pair", any(size >= 14 for size, _, _ in all_pairs))] += 1
        for _, _, payload in test:
            counts[("novel", len(set(payload) - train_colors))] += 1
        if len(train) >= 2:
            for i, (size, bg, payload) in enumerate(train):
                if any(other_size != size and other_bg == bg and other_payload == payload
                       for other_size, other_bg, other_payload in train[:i]):
                    counts[("controlled_scale", True)] += 1
                    break
    _require(len(signatures) == 96, "seeded puzzles should be distinct")
    _require(all(counts[("train_count", n)] > 0 for n in range(2, 11)),
             "automatic train counts did not vary")
    _require(all(counts[("test_count", n)] > 0 for n in range(1, 5)),
             "automatic test counts did not vary")
    _require(counts[("scale_regime", True, False)] >= 10,
             "too few shared-size puzzles")
    _require(counts[("scale_regime", False, True)] >= 10,
             "too few held-out-scale puzzles")
    _require(counts[("source_like", True)] >= 10,
             "too few source-like count and size puzzles")
    _require(palette_sizes == {4, 5, 6, 7, 8},
             "training palette sizes did not span four through eight")
    _require(counts[("backgrounds", True)] >= 20, "background variation absent")
    _require(counts[("zero_payload", True)] >= 10, "zero payload variation absent")
    _require(counts[("repeat_test", True)] >= 8, "repeated-color tests absent")
    _require(counts[("controlled_scale", True)] >= 2,
             "same payload across scales absent")
    _require(counts[("test_palette_variety", True)] >= 10,
             "different test palettes within a puzzle absent")
    _require(counts[("permuted_set", True)] >= 10,
             "shared color sets were not permuted")
    _require(counts[("heldout_and_background", True)] >= 2
             and counts[("repeat_and_background", True)] >= 2,
             "compatible variation axes did not combine")
    _require(counts[("compact_pair", True)] >= 10
             and counts[("large_pair", True)] >= 10,
             "compact and large grids were not both sampled")
    _require(all(counts[("novel", n)] > 0 for n in range(5)),
             "test color novelty did not span zero to four")
    _require(_puzzle_signature(generate_puzzle(123)) ==
             _puzzle_signature(generate_puzzle(123)), "generation is not deterministic")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(4321, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts changed")
    for kwargs in ({"num_train": None, "num_test": 1},
                   {"num_train": None, "num_test": 4},
                   {"num_train": 2, "num_test": None},
                   {"num_train": 10, "num_test": None}):
        puzzle = generate_puzzle(555, **kwargs)
        if kwargs["num_train"] is not None:
            _require(len(puzzle["train"]) == kwargs["num_train"], "partial train count changed")
        if kwargs["num_test"] is not None:
            _require(len(puzzle["test"]) == kwargs["num_test"], "partial test count changed")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
