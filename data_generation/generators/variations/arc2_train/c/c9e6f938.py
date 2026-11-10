import numpy as np


_SOURCE_INPUTS = (
    ((0, 0, 0), (0, 7, 7), (0, 0, 0)),
    ((0, 7, 0), (0, 0, 7), (0, 7, 7)),
    ((0, 0, 0), (7, 0, 0), (0, 0, 0)),
    ((7, 7, 0), (0, 7, 0), (0, 0, 7)),
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_output(grid: np.ndarray) -> np.ndarray:
    return np.concatenate((grid, grid[:, ::-1]), axis=1)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        _grid_signature(input_grid),
        _grid_signature(output_grid),
    )


def _is_original_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for raw_input in _SOURCE_INPUTS:
        source_input = np.asarray(raw_input, dtype=np.int8)
        source_output = _make_output(source_input)
        if (
            np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _validate_optional_count(
    value: int | None,
    name: str,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    return int(value)


def _select_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(num_train, "num_train")
    num_test = _validate_optional_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")

    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity across consecutive seeds.
        num_test = 1 + (seed % 4)
        quotient = seed // 4
        num_train = (
            num_test
            + 1
            + (quotient % (10 - num_test))
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _dimension(rng: np.random.Generator, axis: str, informative: bool,
               exclude: int | None = None) -> int:
    if axis == "width":
        bins = ((1,), (2, 3), tuple(range(4, 9)),
                tuple(range(9, 13)), (13, 14, 15))
        weights = (0.09, 0.20, 0.42, 0.19, 0.10)
    else:
        bins = ((1,), tuple(range(2, 8)), tuple(range(8, 16)),
                tuple(range(16, 24)), tuple(range(24, 31)))
        weights = (0.08, 0.51, 0.20, 0.13, 0.08)
    choices, probabilities = [], []
    for values, weight in zip(bins, weights):
        for value in values:
            if value != exclude and (not informative or value >= 3):
                choices.append(value)
                probabilities.append(weight / len(values))
    probabilities = np.asarray(probabilities)
    probabilities /= probabilities.sum()
    return int(rng.choice(choices, p=probabilities))


def _base_pattern(rng: np.random.Generator, height: int,
                  width: int, style: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    if style == 0:  # scattered
        return rng.random((height, width)) < 0.5
    if style == 1:  # rectangle
        top = int(rng.integers(height))
        bottom = int(rng.integers(top + 1, height + 1))
        left = int(rng.integers(width))
        right = int(rng.integers(left + 1, width + 1))
        return (top <= rr) & (rr < bottom) & (left <= cc) & (cc < right)
    if style == 2:  # stripes
        period = int(rng.integers(2, 6))
        axis = rr if int(rng.integers(2)) else cc
        return (axis + int(rng.integers(period))) % period < int(rng.integers(1, period))
    if style == 3:  # diagonal
        period = int(rng.integers(2, 6))
        sign = 1 if int(rng.integers(2)) else -1
        return (rr + sign * cc + int(rng.integers(period))) % period == 0
    if style == 4:  # checker
        return (rr + cc + int(rng.integers(2))) % 2 == 0
    if style == 5:  # repeated row motif
        return np.tile(rng.random(width) < 0.5, (height, 1))
    mask = np.zeros((height, width), dtype=bool)  # row runs
    for row in range(height):
        left = int(rng.integers(width))
        right = int(rng.integers(left + 1, width + 1))
        mask[row, left:right] = True
    return mask


def _sample_input(rng: np.random.Generator, height: int, width: int,
                  informative: bool) -> np.ndarray:
    area = height * width
    if not informative and rng.random() < 0.065:
        return np.full((height, width), int(rng.integers(10)), dtype=np.int8)
    k = min(area, int(rng.choice((2, 3, 4, 5, 6),
                                 p=(0.51, 0.25, 0.14, 0.07, 0.03))))
    if k == 1:
        return np.full((height, width), int(rng.integers(10)), dtype=np.int8)
    if rng.random() < 0.43:
        colors = np.concatenate(([0], rng.choice(np.arange(1, 10),
                                                   size=k - 1, replace=False))).astype(np.int8)
    else:
        colors = rng.choice(10, size=k, replace=False).astype(np.int8)
    band = int(rng.integers(4))
    lo, hi = ((0.04, 0.22), (0.22, 0.43), (0.43, 0.66),
              (0.66, 0.94))[band]
    count = min(area, max(k - 1, int(round(area * rng.uniform(lo, hi)))))
    base = _base_pattern(rng, height, width, int(rng.integers(7)))
    if rng.random() < 0.5:
        base = ~base
    scores = rng.random((height, width)) + 0.75 * base
    selected = np.argpartition(scores.ravel(), area - count)[area - count:]
    mask = np.zeros(area, dtype=bool)
    mask[selected] = True
    mask = mask.reshape(height, width)
    if height >= 2 and rng.random() < 0.55:
        row, variant = int(rng.integers(height)), int(rng.integers(6))
        if variant == 0:
            mask[row] = False
        elif variant == 1:
            mask[row] = True
        elif variant == 2:
            mask[row] = np.arange(width) % 2 == int(rng.integers(2))
        elif variant == 3:
            mask[row] = False
            mask[row, int(rng.integers(width))] = True
        elif variant == 4:
            mask[row] = mask[row] | mask[row, ::-1]
        else:
            mask[row] = np.roll(mask[row], int(rng.integers(width)))
    grid = np.full((height, width), colors[0], dtype=np.int8)
    locations = np.flatnonzero(mask)
    if len(locations) and k > 1:
        grid.flat[locations] = rng.choice(colors[1:], size=len(locations))
        for color, location in zip(colors[1:], rng.permutation(locations)):
            grid.flat[int(location)] = color
    if informative:
        # The first demonstration exposes asymmetric mirroring and row order.
        grid[0, 0], grid[0, -1] = colors[0], colors[1]
        grid[1] = colors[0]
        grid[1, width // 2] = colors[1]
        if np.array_equal(grid[0], grid[1]):
            grid[1] = colors[0]
    return grid


def _cropped_structure(grid: np.ndarray) -> tuple:
    values, counts = np.unique(grid, return_counts=True)
    mask = grid != values[int(np.argmax(counts))]
    if not np.any(mask):
        return ("uniform",)
    rows, columns = np.nonzero(mask)
    crop = mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    return (crop.shape, tuple(bool(value) for value in crop.flat))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _select_counts(seed, rng, num_train, num_test)
    # The source rule fixes the sole convention: reflection is appended on the right.
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_train_structures = set()
    first_shape = None
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            informative = split == "train" and index == 0
            exclude_h = first_shape[0] if split == "train" and index == 1 else None
            exclude_w = first_shape[1] if split == "train" and index == 1 else None
            familiar_source = None
            if split == "test" and rng.random() < 0.24:
                familiar_source = puzzle["train"][int(rng.integers(num_train))]["input"]
            for _ in range(64):
                height = _dimension(rng, "height", informative, exclude_h)
                width = (familiar_source.shape[1] if familiar_source is not None
                         else _dimension(rng, "width", informative, exclude_w))
                inp = _sample_input(rng, height, width, informative)
                if familiar_source is not None:
                    inp[int(rng.integers(height))] = familiar_source[
                        int(rng.integers(familiar_source.shape[0]))]
                out = _make_output(inp)
                key = _grid_signature(inp)
                structure = _cropped_structure(inp)
                if (key not in used_inputs and not _is_original_pair(inp, out)
                        and (split != "train" or structure not in used_train_structures)):
                    break
            else:
                raise RuntimeError("could not construct a unique input")
            used_inputs.add(key)
            if split == "train":
                used_train_structures.add(structure)
            if informative:
                first_shape = inp.shape
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle needs exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
           "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
               "pair needs exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                   f"{name} must be a two-dimensional NumPy array")
            _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                   f"invalid {name} dimensions")
            _check(np.issubdtype(grid.dtype, np.integer), f"{name} must be integer")
            _check(bool(np.all((grid >= 0) & (grid <= 9))), f"invalid {name} colors")
        inp, out = pair["input"], pair["output"]
        h, w = inp.shape
        _check(out.shape == (h, 2 * w), "wrong output dimensions")
        _check(np.array_equal(out[:, :w], inp), "left half is not copied")
        _check(np.array_equal(out[:, w:], inp[:, ::-1]), "right half is not reflected")
        input_key, pair_key = _grid_signature(inp), _pair_signature(inp, out)
        _check(input_key not in seen_inputs and pair_key not in seen_pairs,
               "duplicate input or pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        _check(not _is_original_pair(inp, out), "exact source pair copied")
    # There are no sampled puzzle-level color roles or hidden parameters.
    _check(any(grid.shape[0] >= 2 and grid.shape[1] >= 2
               and any(not np.array_equal(row, row[::-1]) for row in grid)
               and any(not np.array_equal(grid[0], row) for row in grid[1:])
               for grid in (pair["input"] for pair in train)),
           "training must show an asymmetric row and distinct rows")
    _check(len({pair["input"].shape[0] for pair in train}) >= 2
           and len({pair["input"].shape[1] for pair in train}) >= 2,
           "training must vary height and width")
    _check(len({_cropped_structure(pair["input"]) for pair in train}) == len(train),
           "training repeats a color-normalized cropped layout")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, _pair_signature(pair["input"], pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _predict_from_demonstrations(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check sees only demonstrations and the test input. The right-hand
    # placement is inferred from an asymmetric training row.
    asymmetric = False
    for pair in train:
        inp, out = pair["input"], pair["output"]
        width = inp.shape[1]
        for row, result in zip(inp, out):
            if not np.array_equal(row, row[::-1]):
                asymmetric = True
            _check(list(result) == list(row) + list(row[::-1]),
                   "training contradicts row reflection")
        _check(out.shape[1] == 2 * width, "training contradicts width doubling")
    _check(asymmetric, "training cannot distinguish reversal from repetition")
    return np.asarray([list(row) + list(row[::-1]) for row in test_input],
                      dtype=np.int8)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_puzzle_signature(p) for p in puzzles}) == 42,
           "different seeds repeated a puzzle")
    _check(len({len(p["train"]) for p in puzzles}) >= 4
           and len({len(p["test"]) for p in puzzles}) >= 3,
           "automatic counts lack diversity")
    _check(_puzzle_signature(generate_puzzle(123)) ==
           _puzzle_signature(generate_puzzle(123)), "nondeterministic puzzle")
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4)), 100):
        puzzle = generate_puzzle(seed, *counts)
        _check((len(puzzle["train"]), len(puzzle["test"])) == counts,
               "explicit counts ignored")
    for seed, arguments in enumerate(((None, 1), (None, 4),
                                      (2, None), (10, None)), 200):
        puzzle = generate_puzzle(seed, *arguments)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])),
                                     arguments):
            _check(requested is None or actual == requested, "partial count ignored")
    for arguments in ((1, None), (11, None), (None, 0), (None, 5),
                      (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    grids = {split: [pair["input"] for puzzle in puzzles for pair in puzzle[split]]
             for split in ("train", "test")}
    shapes = {grid.shape for split in grids for grid in grids[split]}
    _check(len(shapes) >= 85 and len({s[1] for s in shapes}) >= 12
           and len({s[0] for s in shapes}) >= 19, "sizes lack diversity")
    _check(len({_cropped_structure(grid) for split in grids
                for grid in grids[split]}) >= 200,
           "color-normalized cropped layouts lack diversity")
    _check(any(g.shape[0] == 1 for g in grids["test"])
           and any(g.shape[1] == 1 for g in grids["test"]),
           "one-row or one-column tests missing")
    _check(any(g.shape[0] == 30 for split in grids for g in grids[split])
           and any(g.shape[1] == 15 for split in grids for g in grids[split]),
           "legal size extremes missing")
    _check(all(len({pair["input"].shape[0] for pair in puzzle["train"]}) > 1
               and len({pair["input"].shape[1] for pair in puzzle["train"]}) > 1
               for puzzle in puzzles), "dimensions stay fixed within a puzzle")
    for split in ("train", "test"):
        group = grids[split]
        _check(sum(len(np.unique(g)) >= 3 for g in group) >= 10,
               f"{split} lacks multicolor grids")
        _check(sum(len(np.unique(g)) == 1 for g in group) >= 2,
               f"{split} lacks uniform grids")
        def occupancy(grid: np.ndarray) -> float:
            return 1 - float(np.unique(grid, return_counts=True)[1].max()) / grid.size
        _check(sum(occupancy(g) <= 0.25 for g in group) >= 20,
               f"{split} lacks sparse grids")
        _check(sum(occupancy(g) >= 0.45 for g in group) >= 20,
               f"{split} lacks balanced or dense grids")
        _check(any(any(np.all(row == row[0]) for row in g) for g in group),
               f"{split} lacks solid rows")
        _check(any(any(np.array_equal(row, row[::-1]) for row in g) for g in group),
               f"{split} lacks palindromic rows")
        _check(sum(len({tuple(row) for row in g}) < len(g) for g in group) >= 10,
               f"{split} lacks repeated rows")
        _check(sum(any(np.all(row == np.unique(g, return_counts=True)[0][
            np.argmax(np.unique(g, return_counts=True)[1])]) for row in g)
            for g in group) >= 20, f"{split} lacks blank rows")
        _check(sum(any(np.all(row != np.unique(g, return_counts=True)[0][
            np.argmax(np.unique(g, return_counts=True)[1])]) for row in g)
            for g in group) >= 15, f"{split} lacks full rows")
        _check(sum(len(np.unique(g)) >= 3 and
                   (g.shape[0] >= 10 or g.shape[1] >= 9) for g in group) >= 20,
               f"{split} lacks combined size and color variation")
    _check(sum(len({tuple(np.unique(pair["input"]))
                    for pair in p["train"] + p["test"]}) > 1 for p in puzzles) >= 35,
           "within-puzzle palettes lack variation")
    denser_train = sum(
        np.mean([occupancy(pair["input"]) for pair in p["train"]]) >
        np.mean([occupancy(pair["input"]) for pair in p["test"]])
        for p in puzzles)
    _check(10 <= denser_train <= 32,
           "training and testing retain a fixed density direction")
    familiar_tests = 0
    new_tests = 0
    for puzzle in puzzles:
        known_rows = {tuple(row) for pair in puzzle["train"] for row in pair["input"]}
        for pair in puzzle["test"]:
            if any(tuple(row) in known_rows for row in pair["input"]):
                familiar_tests += 1
            else:
                new_tests += 1
    _check(familiar_tests >= 10 and new_tests >= 40,
           "test rows are always familiar or always new")
    _check(sum(g.shape[0] <= 4 and g.shape[1] <= 4 and
               len(np.unique(g)) == 2 and occupancy(g) <= 0.33
               for split in grids for g in grids[split]) >= 4,
           "source-like compact sparse binary grids missing")
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            _check(all(list(out[r]) == list(inp[r]) + list(inp[r][::-1])
                       for r in range(inp.shape[0])), "incorrect row reflection")
        for pair in puzzle["test"]:
            prediction = _predict_from_demonstrations(
                [{"input": item["input"].copy(), "output": item["output"].copy()}
                 for item in puzzle["train"]], pair["input"].copy())
            _check(np.array_equal(prediction, pair["output"]),
                   "test output not inferred from demonstrations and input")
