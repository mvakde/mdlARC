"""Generate aligned two-panel Boolean-union ARC puzzles."""

import numpy as np


_SOURCE_PAIRS = {
    (("3300", "3000", "0003", "0033", "3000", "3333",
      "0500", "5500", "5050", "5555", "5550", "5050"),
     ("4400", "4400", "4044", "4444", "4440", "4444")),
    (("3303", "3033", "0300", "0030", "3030", "0003",
      "5000", "0555", "5005", "0555", "5550", "5000"),
     ("4404", "4444", "4404", "0444", "4440", "4004")),
    (("3300", "0333", "3303", "0330", "3030", "3000",
      "0555", "5555", "5550", "5555", "5000", "0550"),
     ("4444", "4444", "4444", "4444", "4040", "4440")),
    (("0333", "0303", "0000", "3033", "3000", "0303",
      "0505", "0000", "0000", "5050", "5000", "5505"),
     ("0444", "0404", "0000", "4044", "4000", "4404")),
    (("3303", "0303", "0003", "3303", "3033", "0333",
      "0000", "5005", "0050", "5005", "5555", "5500"),
     ("4404", "4404", "0044", "4404", "4444", "4444")),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _sample_shape(rng):
    """Four overlapping aspect-ratio ranges, all with a legal stacked input."""
    kind = int(rng.choice(4, p=(0.23, 0.37, 0.20, 0.20)))
    if kind == 0:  # short and wide
        return int(rng.integers(2, 5)), int(rng.integers(10, 26))
    if kind == 1:  # familiar compact panels, including the source size
        return int(rng.integers(4, 11)), int(rng.integers(3, 13))
    if kind == 2:  # tall panels
        return int(rng.integers(11, 16)), int(rng.integers(2, 10))
    return int(rng.integers(8, 16)), int(rng.integers(12, 26))


def _scores(rng, height, width, style):
    noise = rng.random((height, width))
    rows, cols = np.indices((height, width), dtype=float)
    if style == 0:  # scattered
        return noise
    if style == 1:  # horizontal bands
        return rng.random((height, 1)) + 0.12 * noise
    if style == 2:  # vertical bands
        return rng.random((1, width)) + 0.12 * noise
    if style == 3:  # connected soft fragments
        field = np.full((height, width), -1.0e9)
        for _ in range(int(rng.integers(1, 5))):
            r = rng.uniform(0, height - 1)
            c = rng.uniform(0, width - 1)
            sr = rng.uniform(0.8, max(1.1, height / 2))
            sc = rng.uniform(0.8, max(1.1, width / 2))
            field = np.maximum(field, -((rows - r) / sr) ** 2 - ((cols - c) / sc) ** 2)
        return field + 0.08 * noise
    if style == 4:  # diagonal bands
        period = int(rng.integers(2, max(3, min(height, width) + 2)))
        phase = int(rng.integers(0, period))
        diagonal = rows + cols if int(rng.integers(0, 2)) else rows - cols
        residue = np.mod(diagonal + phase, period)
        return -np.minimum(residue, period - residue) + 0.1 * noise
    # Composed rectangles
    field = 0.1 * noise
    for _ in range(int(rng.integers(1, 5))):
        r0 = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width))
        r1 = int(rng.integers(r0 + 1, height + 1))
        c1 = int(rng.integers(c0 + 1, width + 1))
        field[r0:r1, c0:c1] += rng.uniform(0.7, 1.5)
    return field


def _select(scores, count, eligible=None):
    candidates = (np.arange(scores.size) if eligible is None
                  else np.flatnonzero(eligible.ravel()))
    selected = np.zeros(scores.size, dtype=bool)
    if count:
        order = np.argsort(scores.ravel()[candidates], kind="stable")
        selected[candidates[order[-count:]]] = True
    return selected.reshape(scores.shape)


def _union_count(rng, cells, anchor):
    if anchor:
        return int(rng.integers(max(3, int(np.ceil(0.40 * cells))),
                                min(cells - 1, int(np.floor(0.72 * cells))) + 1))
    band = int(rng.choice(4, p=(0.24, 0.41, 0.31, 0.04)))
    if band == 3:
        return cells
    bounds = ((0.17, 0.38), (0.39, 0.70), (0.71, 0.96))[band]
    low = max(3, int(np.ceil(bounds[0] * cells)))
    high = max(low, min(cells - 1, int(np.floor(bounds[1] * cells))))
    return int(rng.integers(low, high + 1))


def _relationship(rng, anchor, test):
    if anchor:
        return "balanced"
    names = ("balanced", "high_overlap", "near_identical", "disjoint",
             "contained_top", "contained_bottom", "empty_top", "empty_bottom")
    weights = ((0.29, 0.15, 0.08, 0.13, 0.105, 0.105, 0.07, 0.07)
               if not test else
               (0.20, 0.18, 0.11, 0.16, 0.11, 0.11, 0.065, 0.065))
    return names[int(rng.choice(len(names), p=weights))]


def _masks(rng, shape, anchor, test, two_visible):
    height, width = shape
    cells = height * width
    union_count = _union_count(rng, cells, anchor)
    relation = _relationship(rng, anchor, test)
    if two_visible and relation.startswith("empty"):
        relation = "contained_top" if relation == "empty_bottom" else "contained_bottom"
    if union_count == cells and relation in ("contained_top", "contained_bottom",
                                             "empty_top", "empty_bottom"):
        relation = "balanced"
    if relation == "near_identical" and union_count < 10:
        relation = "high_overlap"
    if relation == "high_overlap" and union_count < 7:
        relation = "balanced"

    union_style = int(rng.integers(0 if not test else 1, 6))
    union = _select(_scores(rng, height, width, union_style), union_count)

    if relation == "disjoint" or relation.startswith("empty"):
        overlap_count = 0
    elif relation == "near_identical":
        overlap_count = int(rng.integers(int(np.ceil(0.80 * union_count)),
                                         union_count - 1))
    elif relation == "high_overlap":
        overlap_count = int(rng.integers(int(np.ceil(0.55 * union_count)),
                                         min(union_count - 2,
                                             int(np.floor(0.79 * union_count))) + 1))
    elif relation.startswith("contained"):
        overlap_count = int(rng.integers(1, union_count))
    else:
        overlap_count = int(rng.integers(1, max(2, int(np.floor(0.46 * union_count)) + 1)))
        overlap_count = min(overlap_count, union_count - 2)

    overlap_style = int(rng.integers(0, 6))
    overlap = _select(_scores(rng, height, width, overlap_style), overlap_count, union)
    exclusive = union & ~overlap
    exclusive_count = int(np.count_nonzero(exclusive))
    if relation == "contained_top" or relation == "empty_bottom":
        top_only_count = exclusive_count
    elif relation == "contained_bottom" or relation == "empty_top":
        top_only_count = 0
    elif relation == "balanced":
        low = max(1, int(np.ceil(0.25 * exclusive_count)))
        high = min(exclusive_count - 1, int(np.floor(0.75 * exclusive_count)))
        top_only_count = int(rng.integers(low, max(low, high) + 1))
    else:
        top_only_count = int(rng.integers(1, exclusive_count))

    allocation_style = int(rng.integers(0, 6))
    top_only = _select(_scores(rng, height, width, allocation_style),
                       top_only_count, exclusive)
    bottom_only = exclusive & ~top_only
    top = overlap | top_only
    bottom = overlap | bottom_only
    return top, bottom


def _palette(rng, mode, base, result, index):
    background, upper, lower = base
    if mode == "fixed":
        return background, upper, lower, result
    if mode == "swap":
        return (background, upper, lower, result) if index % 2 == 0 else (
            background, lower, upper, result)
    if index == 0:
        return background, upper, lower, result
    if index == 1:
        alternatives = [c for c in range(10) if c not in (background, upper, lower, result)]
        return int(rng.choice(alternatives)), lower, upper, result
    colors = [int(c) for c in rng.choice([c for c in range(10) if c != result],
                                         size=3, replace=False)]
    return colors[0], colors[1], colors[2], result


def _render_pair(top, bottom, palette):
    background, upper, lower, result = palette
    height, width = top.shape
    inp = np.full((2 * height, width), background, dtype=np.int8)
    inp[:height][top] = upper
    inp[height:][bottom] = lower
    out = np.full((height, width), background, dtype=np.int8)
    out[top | bottom] = result
    return {"input": inp, "output": out}


def _grid_rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _matches_source(pair):
    return (_grid_rows(pair["input"]), _grid_rows(pair["output"])) in _SOURCE_PAIRS


def _array_key(array):
    return array.shape, np.ascontiguousarray(array).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    result = int(rng.integers(0, 10))
    base = tuple(int(c) for c in rng.choice([c for c in range(10) if c != result],
                                             size=3, replace=False))
    mode = ("fixed", "swap", "varied")[int(rng.choice(3, p=(0.35, 0.20, 0.45)))]
    variable_shape = bool(rng.random() < 0.68)
    base_shape = _sample_shape(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    total = train_count + test_count
    for index in range(total):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        shape = _sample_shape(pair_rng) if variable_shape and index else base_shape
        if variable_shape and index == 1 and shape == base_shape:
            shape = (shape[0] + 1 if shape[0] < 15 else shape[0] - 1, shape[1])
        palette = _palette(pair_rng, mode, base, result, index)
        split = "train" if index < train_count else "test"
        for _ in range(80):
            top, bottom = _masks(pair_rng, shape, index == 0, split == "test",
                                 index == 1 and mode != "fixed")
            pair = _render_pair(top, bottom, palette)
            key = _array_key(pair["input"])
            if key not in seen and not _matches_source(pair):
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not create a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    height, width = out.shape
    _require(inp.shape == (2 * height, width), "input must hold two aligned panels")
    top, bottom = inp[:height], inp[height:]
    top_values = set(int(v) for v in np.unique(top))
    bottom_values = set(int(v) for v in np.unique(bottom))
    shared = top_values & bottom_values
    _require(len(shared) == 1, "input panels must reveal one shared background")
    background = next(iter(shared))
    _require(background in top_values and background in bottom_values,
             "background must be visible in both panels")
    _require(len(top_values) <= 2 and len(bottom_values) <= 2,
             "each panel has at most one foreground color")
    upper = next(iter(top_values - {background}), None)
    lower = next(iter(bottom_values - {background}), None)
    _require(upper is None or lower is None or upper != lower,
             "the input foreground roles must be distinct")
    top_mask, bottom_mask = top != background, bottom != background
    categories = (
        int(np.count_nonzero(~top_mask & ~bottom_mask)),
        int(np.count_nonzero(top_mask & ~bottom_mask)),
        int(np.count_nonzero(~top_mask & bottom_mask)),
        int(np.count_nonzero(top_mask & bottom_mask)),
    )
    _require(categories[1] + categories[2] + categories[3] > 0,
             "at least one layer must contribute")
    return background, upper, lower, top_mask, bottom_mask, categories


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a two-dimensional NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape),
                         f"{name} has illegal dimensions")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         bool(np.all((grid >= 0) & (grid <= 9))),
                         f"{name} must contain integer ARC colors")
            input_key = _array_key(pair["input"])
            pair_key = input_key, _array_key(pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            _require(not _matches_source(pair), "exact source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            facts[split].append(_analyze_pair(pair))

    train_results = set()
    for pair, fact in zip(puzzle["train"], facts["train"]):
        background = fact[0]
        train_results.update(int(c) for c in np.unique(pair["output"]) if c != background)
    _require(len(train_results) == 1, "training must reveal one constant result color")
    result = next(iter(train_results))
    _require(any(all(n > 0 for n in fact[5]) for fact in facts["train"]),
             "training needs a complete four-case demonstration")

    train_backgrounds = {fact[0] for fact in facts["train"]}
    train_shapes = {pair["output"].shape for pair in puzzle["train"]}
    train_foregrounds = {(fact[1], fact[2]) for fact in facts["train"]}
    for split in ("train", "test"):
        for pair, fact in zip(puzzle[split], facts[split]):
            background, upper, lower, top, bottom, _ = fact
            _require(background != result and upper != result and lower != result,
                     "result color must be distinct from input roles")
            expected = np.where(top | bottom, result, background)
            _require(np.array_equal(pair["output"], expected),
                     "output is not the aligned Boolean union")
            _require(set(int(c) for c in np.unique(pair["output"])) <=
                     {background, result}, "output has an extra color")
            if split == "test":
                if pair["output"].shape not in train_shapes:
                    _require(len(train_shapes) >= 2,
                             "a novel test size needs varied training sizes")
                if background not in train_backgrounds:
                    _require(len(train_backgrounds) >= 2,
                             "a novel test background needs varied training backgrounds")
                if upper is not None and lower is not None and (upper, lower) not in train_foregrounds:
                    _require(len(train_foregrounds) >= 2,
                             "novel test foreground roles need varied training roles")


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def _predict_from_demonstrations(train, test_input):
    results = set()
    for pair in train:
        out = pair["output"]
        height = out.shape[0]
        shared = set(int(v) for v in np.unique(pair["input"][:height])) & \
                 set(int(v) for v in np.unique(pair["input"][height:]))
        _require(len(shared) == 1, "demonstration background is ambiguous")
        bg = next(iter(shared))
        results.update(int(v) for v in np.unique(out) if v != bg)
    _require(len(results) == 1, "demonstrations do not identify result color")
    height = test_input.shape[0] // 2
    shared = set(int(v) for v in np.unique(test_input[:height])) & \
             set(int(v) for v in np.unique(test_input[height:]))
    _require(len(shared) == 1, "test background is ambiguous")
    bg = next(iter(shared))
    return np.where((test_input[:height] != bg) | (test_input[height:] != bg),
                    next(iter(results)), bg)


def self_test() -> None:
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None), (6, 3), (None, None)):
        puzzle = generate_puzzle(503 + len(str(counts)), *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "train count ignored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "test count ignored")
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(503 + len(str(counts)), *counts)),
                 "generation is nondeterministic")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2.0, 1), (2, True)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")

    corpus = [generate_puzzle(seed) for seed in range(1000, 1042)]
    train_counts, test_counts, shapes, palettes, structures = set(), set(), set(), set(), set()
    relations = set()
    variable_shapes = variable_backgrounds = variable_foregrounds = swapped_train_roles = 0
    reverse_densities = dense_trains = sparse_tests = full_unions = 0
    balanced_tests = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_shapes, local_backgrounds, local_foregrounds = set(), set(), set()
        train_densities, test_densities = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                fact = _analyze_pair(pair)
                bg, upper, lower, top, bottom, categories = fact
                shape = pair["output"].shape
                density = 1 - categories[0] / (shape[0] * shape[1])
                shapes.add(shape)
                palettes.add((bg, upper, lower))
                local_shapes.add(shape)
                local_backgrounds.add(bg)
                local_foregrounds.add((upper, lower))
                structures.add((shape, top.tobytes(), bottom.tobytes()))
                if split == "train":
                    train_densities.append(density)
                else:
                    test_densities.append(density)
                    prediction = _predict_from_demonstrations(puzzle["train"], pair["input"].copy())
                    _require(np.array_equal(prediction, pair["output"]),
                             "training-only prediction disagrees with test output")
                    balanced_tests += categories[1] >= 0.20 * sum(categories[1:]) and \
                                      categories[2] >= 0.20 * sum(categories[1:])
                full_unions += categories[0] == 0
                if categories[3] == 0:
                    relations.add("disjoint")
                if (categories[1] == 0 or categories[2] == 0) and np.any(top) and np.any(bottom):
                    relations.add("contained")
                if not np.any(top) or not np.any(bottom):
                    relations.add("empty")
                union = sum(categories[1:])
                if categories[3] / union >= 0.55:
                    relations.add("high_overlap")
                if categories[3] / union >= 0.80:
                    relations.add("near_identical")
        variable_shapes += len(local_shapes) > 1
        variable_backgrounds += len(local_backgrounds) > 1
        variable_foregrounds += len(local_foregrounds) > 1
        first = _analyze_pair(puzzle["train"][0])
        swapped_train_roles += any(
            _analyze_pair(pair)[1:3] == (first[2], first[1])
            for pair in puzzle["train"][1:]
        )
        reverse_densities += max(train_densities) > min(test_densities)
        dense_trains += max(train_densities) >= 0.75
        sparse_tests += min(test_densities) <= 0.55
    _require(len({_puzzle_key(p) for p in corpus}) >= 40, "too few distinct puzzles")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity is too low")
    _require(len(shapes) >= 40 and len(palettes) >= 40 and len(structures) >= 200,
             "shape, palette, or structural diversity is too low")
    _require(any(h <= 4 and w >= 10 for h, w in shapes) and
             any(h >= 11 for h, _ in shapes) and any(h * w >= 180 for h, w in shapes),
             "short, tall, and large panels must all occur")
    _require(variable_shapes >= 15 and variable_backgrounds >= 10 and
             variable_foregrounds >= 15 and swapped_train_roles >= 15,
             "within-puzzle variation is too low")
    _require(reverse_densities >= 8 and dense_trains >= 8 and sparse_tests >= 8,
             "train/test density ranges are still too constrained")
    _require(full_unions >= 2 and balanced_tests >= 5,
             "full unions or substantial two-layer tests are missing")
    _require(relations == {"disjoint", "contained", "empty", "high_overlap",
                             "near_identical"}, "mask relationships are too narrow")
