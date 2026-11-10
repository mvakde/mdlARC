"""Horizontal-run counting puzzles with fixed 3, 2, and 0 output roles."""

import numpy as np


_FIELDS = (0, 1, 4, 5, 6, 7, 8, 9)
_SIGNATURES = tuple(
    (threes, twos)
    for total in range(2, 7)
    for threes in range(total + 1)
    for twos in (total - threes,)
)
_SOURCE_INPUTS = {
    (
        "11111111111111", "11333311111111", "11111111111111",
        "11111111122211", "11111111111111", "33311111111111",
        "11111111111111", "11111111333311", "12222211111111",
        "11111111111111",
    ),
    (
        "5555555555", "3333555555", "5555555555", "5225555555",
        "5555552225", "5555555555", "5555555555", "5555333355",
        "5555555555", "5555555555",
    ),
    (
        "888888888888888", "888888888888888", "882222888888888",
        "888888888888888", "888888833333888", "888888888888888",
        "883333338888888", "888888888888888", "888888888222228",
        "888888888888888", "822288888888888", "888888882222888",
        "888888888888888",
    ),
    (
        "8888888888888888888", "3333338888882222888",
        "8888888888888888888", "8833333333888888888",
        "8888888888888888888", "8888888888888833338",
        "8888833388888888888", "8888888888888888888",
        "8888888888888888888", "8888888888888888888",
    ),
    (
        "0000000000", "0333300000", "0000000000", "0022222200",
        "0000000000", "0000000000", "0000000000", "0000033330",
        "0000000000", "0000022000", "0000000000", "0330000000",
        "0000000000", "0000222200", "0000000000",
    ),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _optional_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _optional_count("num_train", num_train, 2, 10)
    num_test = _optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        choices = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        return choices[int(rng.integers(len(choices)))]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _shape(rng, long_bar=False):
    """Sample all three aspect regimes on either side of the split."""
    mode = int(rng.integers(3))
    if mode == 0:  # wide
        height = int(rng.integers(7 if long_bar else 5, 18))
        width = int(rng.integers(max(14 if long_bar else 8, height + 3), 31))
    elif mode == 1:  # near square
        height = int(rng.integers(12 if long_bar else 8, 25))
        width = int(rng.integers(max(14 if long_bar else 8, height - 2), min(30, height + 2) + 1))
    else:  # tall
        width = int(rng.integers(14 if long_bar else 8, 21))
        height = int(rng.integers(max(16 if long_bar else 11, width + 3), 31))
    return height, width


def _choose_rows(rng, height, count):
    base = np.sort(rng.choice(height - count + 1, size=count, replace=False))
    rows = [int(value) + index for index, value in enumerate(base)]
    # Shifting an entire row set preserves separation and deliberately reaches edges.
    if int(rng.integers(4)) == 0:
        delta = -rows[0]
    elif int(rng.integers(4)) == 0:
        delta = height - 1 - rows[-1]
    else:
        delta = 0
    return [row + delta for row in rows]


def _groups(rng, total, height, width, shared=False, triple=False, anchor=False):
    maximum = min(total, (height + 1) // 2)
    minimum = max(1, (total + 2) // 3)
    if anchor:
        count = total - 1  # one pair and singleton rows for length contrast
    elif triple and total >= 3 and total - 2 <= maximum:
        count = total - 2
    else:
        if shared and total > 1:
            maximum = min(maximum, total - 1)
        count = int(rng.integers(minimum, maximum + 1))
    _require(minimum <= count <= maximum, "shape cannot hold separated runs")
    sizes = [1] * count
    extra = total - count
    if triple and extra >= 2 and not anchor:
        sizes[int(rng.integers(count))] = 3
        extra -= 2
    while extra:
        choices = [i for i, size in enumerate(sizes) if size < 3]
        chosen = choices[int(rng.integers(len(choices)))]
        sizes[chosen] += 1
        extra -= 1
    rng.shuffle(sizes)
    _require(max(sizes) * 2 - 1 <= width, "row too narrow for its groups")
    return sizes


def _pick_length(rng, maximum):
    # A uniform length pool makes long and short runs common, independent of color.
    return int(rng.integers(1, min(15, maximum) + 1))


def _make_pair(rng, signature, background, long_and_short=False, inversion=False,
               shared=False, triple=False):
    count_3, count_2 = signature
    total = count_3 + count_2
    height, width = _shape(rng, long_and_short)
    sizes = _groups(rng, total, height, width, shared, triple, long_and_short)
    rows = _choose_rows(rng, height, len(sizes))
    colors = [3] * count_3 + [2] * count_2
    rng.shuffle(colors)
    if inversion:
        colors.remove(2)
        colors.insert(0, 2)
    lengths = [None] * total
    starts = np.cumsum([0] + sizes[:-1]).tolist()
    if long_and_short:
        singleton = next(start for start, size in zip(starts, sizes) if size == 1)
        lengths[singleton] = int(rng.integers(8, min(15, width) + 1))
        short = next(i for i in range(total) if i != singleton)
        lengths[short] = 1
    grid = np.full((height, width), background, dtype=np.int8)
    for row, start, size in zip(rows, starts, sizes):
        group = list(range(start, start + size))
        budget = width - size + 1
        order = group.copy()
        rng.shuffle(order)
        for index in order:
            if lengths[index] is None:
                reserved = sum(lengths[j] if lengths[j] is not None else 1 for j in group if j != index)
                lengths[index] = _pick_length(rng, budget - reserved)
        occupied = sum(lengths[j] for j in group) + size - 1
        slack = width - occupied
        gaps = rng.multinomial(slack, np.full(size + 1, 1 / (size + 1)))
        column = int(gaps[0])
        for local, index in enumerate(group):
            length = lengths[index]
            grid[row, column:column + length] = colors[index]
            column += length
            if local + 1 < size:
                column += 1 + int(gaps[local + 1])
    output = np.zeros((2, 3), dtype=np.int8)
    output.flat[:count_3] = 3
    output.flat[count_3:total] = 2
    return {"input": grid, "output": output}


def _input_key(grid):
    return grid.shape, tuple(int(cell) for cell in grid.flat)


def _text_rows(grid):
    return tuple("".join(str(int(cell)) for cell in row) for row in grid)


def _inspect(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    field = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[field]) == 1, "field must dominate uniquely")
    _require(field in _FIELDS, "field conflicts with foreground")
    _require(set(np.unique(grid)).issubset({field, 2, 3}), "unrelated input color")
    foreground = grid != field
    _require(not np.any(foreground[:-1] & foreground[1:]), "runs touch vertically")
    runs = []
    for row in range(grid.shape[0]):
        col = 0
        while col < grid.shape[1]:
            if grid[row, col] == field:
                col += 1
                continue
            start = col
            color = int(grid[row, col])
            while col < grid.shape[1] and grid[row, col] == color:
                col += 1
            _require(col == grid.shape[1] or grid[row, col] == field,
                     "different bars require a field-colored gap")
            _require(1 <= col - start <= 15, "bar length outside generated family")
            runs.append((row, start, col, color))
    _require(2 <= len(runs) <= 6, "run count outside output capacity")
    return field, runs


def _signature(runs):
    return sum(run[3] == 3 for run in runs), sum(run[3] == 2 for run in runs)


def _expected_output(runs):
    a, b = _signature(runs)
    output = np.zeros((2, 3), dtype=np.int8)
    output.flat[:a] = 3
    output.flat[a:a + b] = 2
    return output


def _training_evidence(infos):
    _require(any(0 < a and 0 < b and a + b < 6 for _, runs in infos
                 for a, b in [_signature(runs)]), "training needs a padded mixed example")
    _require(any(a > 0 and b > 0 and a + b == 6 for _, runs in infos
                 for a, b in [_signature(runs)]), "training needs a full mixed example")
    _require(len({field for field, _ in infos}) >= 2, "training needs varied fields")
    all_runs = [run for _, runs in infos for run in runs]
    _require(any(run[2] - run[1] == 1 for run in all_runs), "training needs a one-cell bar")
    _require(any(run[2] - run[1] >= 8 for run in all_runs), "training needs a long bar")
    _require(any(len({run[0] for run in runs}) < len(runs) for _, runs in infos),
             "training needs a shared row")
    _require(any(runs[0][3] == 2 and any(run[3] == 3 for run in runs[1:])
                 for _, runs in infos), "training needs spatial color inversion")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The source establishes literal 3-before-2 and output-empty 0. Those
    # conventions stay fixed; field color and all geometry vary by pair.
    fields = list(rng.permutation(_FIELDS))
    padded_total = int(rng.integers(4, 6))
    padded_first = int(rng.integers(1, padded_total))
    full_first = int(rng.integers(1, 6))
    train_signatures = [(padded_first, padded_total - padded_first),
                        (full_first, 6 - full_first)]
    for _ in range(2, train_count):
        options = [s for s in _SIGNATURES if s not in train_signatures]
        train_signatures.append(options[int(rng.integers(len(options)))])
    test_signatures = []
    for index in range(test_count):
        if index == 0 and int(rng.integers(4)) != 0:
            options = [s for s in _SIGNATURES if s not in train_signatures]
        elif index == 1 and int(rng.integers(2)) == 0:
            options = [s for s in train_signatures if s not in test_signatures]
        else:
            options = [s for s in _SIGNATURES if s not in test_signatures]
        test_signatures.append(options[int(rng.integers(len(options)))])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, signatures in (("train", train_signatures), ("test", test_signatures)):
        for index, signature in enumerate(signatures):
            background = (int(fields[index]) if split == "train" and index < 2
                          else int(rng.choice(_FIELDS)))
            for _ in range(20):
                pair = _make_pair(
                    rng, signature, background,
                    long_and_short=(split == "train" and index == 0),
                    inversion=(split == "train" and index == 0),
                    shared=(split == "test" and int(rng.integers(4)) != 0),
                    triple=(int(rng.integers(4)) == 0),
                )
                key = _input_key(pair["input"])
                if key not in seen and _text_rows(pair["input"]) not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("could not construct a unique input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test only")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen = set()
    train_infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output only")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer), "grid must be a 2D integer array")
                _require(all(1 <= dim <= 30 for dim in grid.shape)
                         and np.all((0 <= grid) & (grid <= 9)), "grid out of ARC bounds")
            inp, out = pair["input"], pair["output"]
            key = _input_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_text_rows(inp) not in _SOURCE_INPUTS, "source input copied")
            field, runs = _inspect(inp)
            _require(out.shape == (2, 3) and np.array_equal(out, _expected_output(runs)),
                     "output must pack 3-runs, then 2-runs, then zeroes")
            if split == "train":
                train_infos.append((field, runs))
    _training_evidence(train_infos)


def _fingerprint(puzzle):
    return tuple((_input_key(pair["input"]), _input_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                     (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(7000 + offset, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "train count ignored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    a = generate_puzzle(314159, 7, 3)
    _require(_fingerprint(a) == _fingerprint(generate_puzzle(314159, 7, 3)),
             "same seed changed puzzle")
    _require(_fingerprint(a) != _fingerprint(generate_puzzle(314160, 7, 3)),
             "adjacent seeds duplicated puzzle")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    _require(len({_fingerprint(p) for p in puzzles}) >= 40, "puzzles lack diversity")
    layouts, shapes, signatures, fields, lengths = set(), set(), set(), set(), set()
    test_signatures, test_shared, test_triples = set(), 0, 0
    aspect = {"train": set(), "test": set()}
    edge_rows = 0
    review = {split: {name: 0 for name in (
        "same_color_shared", "one_per_row", "one_cell", "long_eight",
        "length_contrast", "padded", "full", "one_color", "field_zero",
    )} for split in ("train", "test")}
    output_patterns = {"train": set(), "test": set()}
    fields_by_split = {"train": set(), "test": set()}
    familiar_tests = novel_tests = multi_test_variety = multi_test_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        known_signatures = {_signature(_inspect(pair["input"])[1])
                            for pair in puzzle["train"]}
        current_test_signatures = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                field, runs = _inspect(inp)
                _require(np.array_equal(pair["output"], _expected_output(runs)),
                         "independent run count differs from answer")
                shapes.add(inp.shape)
                fields.add((split, field))
                signature = _signature(runs)
                signatures.add(signature)
                fields_by_split[split].add(field)
                output_patterns[split].add(tuple(int(x) for x in pair["output"].flat))
                aspect[split].add("wide" if inp.shape[1] >= inp.shape[0] + 3 else
                                  "tall" if inp.shape[0] >= inp.shape[1] + 3 else "square")
                layout = tuple((r, c, e - c, color) for r, c, e, color in runs)
                layouts.add((inp.shape, layout))
                pair_lengths = [e - c for _, c, e, _ in runs]
                lengths.update(pair_lengths)
                edge_rows += any(r in (0, inp.shape[0] - 1) for r, _, _, _ in runs)
                row_counts = [sum(other[0] == run[0] for other in runs) for run in runs]
                review[split]["one_per_row"] += max(row_counts) == 1
                review[split]["same_color_shared"] += any(
                    sum(other[0] == row and other[3] == color for other in runs) >= 2
                    for row in {run[0] for run in runs} for color in (2, 3))
                review[split]["one_cell"] += 1 in pair_lengths
                review[split]["long_eight"] += max(pair_lengths) >= 8
                review[split]["length_contrast"] += max(pair_lengths) - min(pair_lengths) >= 6
                review[split]["padded"] += sum(signature) < 6
                review[split]["full"] += sum(signature) == 6
                review[split]["one_color"] += 0 in signature
                review[split]["field_zero"] += field == 0
                if split == "test":
                    current_test_signatures.add(signature)
                    familiar_tests += signature in known_signatures
                    novel_tests += signature not in known_signatures
                    test_signatures.add(signature)
                    max_row_count = max(row_counts)
                    test_shared += max_row_count >= 2
                    test_triples += max_row_count >= 3
        if len(puzzle["test"]) >= 2:
            multi_test_puzzles += 1
            multi_test_variety += len(current_test_signatures) >= 2
    _require(len(layouts) >= 150 and len(shapes) >= 30, "layouts or shapes lack diversity")
    _require(len(signatures) >= 15 and len(test_signatures) >= 12,
             "run count combinations lack diversity")
    _require(all(len(aspect[split]) == 3 for split in aspect), "aspect regimes missing")
    _require(len(fields) >= 12 and set(range(1, 9)).issubset(lengths)
             and max(lengths) >= 12, "field or bar length range too narrow")
    _require(test_shared >= 10 and test_triples >= 2 and edge_rows >= 10,
             "shared, triple, or edge placements absent")
    _require(multi_test_puzzles == multi_test_variety and familiar_tests >= 5
             and novel_tests >= 10, "test count regimes lack within- or cross-puzzle variety")
    _require(len(output_patterns["test"]) >= 12 and len(test_signatures) >= 12,
             "test answers repeat too often")
    for split in ("train", "test"):
        item = review[split]
        _require(item["same_color_shared"] >= 5 and item["one_per_row"] >= 1,
                 f"{split} same-color or one-per-row layouts missing")
        _require(item["one_cell"] >= 5 and item["long_eight"] >= 5
                 and item["length_contrast"] >= 5,
                 f"{split} bar lengths lack range")
        _require(item["padded"] >= 5 and item["full"] >= 5
                 and item["one_color"] >= 5, f"{split} count regimes missing")
        _require(item["field_zero"] >= 2 and fields_by_split[split] == set(_FIELDS),
                 f"{split} field-color regimes missing")
