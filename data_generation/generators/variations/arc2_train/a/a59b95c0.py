import numpy as np


_SOURCE_INPUTS = {
    (7, 7, 7, 7, 2, 2, 7, 7, 2),
    (3, 4, 4, 3, 3, 3, 3, 4, 4),
    (8, 2, 1, 1, 8, 3, 2, 1, 3),
    (2, 3, 2, 3, 3, 2, 2, 2, 1),
    (9, 7, 9, 9, 6, 7, 7, 6, 6),
    (4, 3, 2, 2, 1, 4, 3, 1, 2),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _key(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _layout(grid):
    labels = {}
    values = []
    for raw in grid.flat:
        value = int(raw)
        if value not in labels:
            labels[value] = len(labels)
        values.append(labels[value])
    return tuple(grid.shape), tuple(values)


def _shape_choices(k):
    return [(h, w) for h in range(2, 7) for w in range(2, 7)
            if h * w >= k and k * h <= 30 and k * w <= 30]


def _choose_shape(rng, k, excluded=()):
    choices = [s for s in _shape_choices(k) if s not in excluded]
    if not choices:
        choices = _shape_choices(k)
    return choices[int(rng.integers(len(choices)))]


def _positive_counts(rng, cells, k, style):
    if style == "balanced":
        counts = np.full(k, cells // k, dtype=int)
        counts[:cells % k] += 1
    elif style == "skew":
        counts = np.ones(k, dtype=int)
        counts[0] = cells - k + 1
    else:
        cuts = np.sort(rng.choice(np.arange(1, cells), size=k - 1, replace=False))
        counts = np.diff(np.concatenate(([0], cuts, [cells])))
    rng.shuffle(counts)
    return counts


def _paint(rng, shape, palette, style, motif):
    h, w = shape
    counts = _positive_counts(rng, h * w, len(palette), style)
    if motif != "scatter":
        counts = np.sort(counts)[::-1]
    colors = np.repeat(np.asarray(palette, dtype=np.int8), counts)
    positions = [(r, c) for r in range(h) for c in range(w)]
    if motif == "scatter":
        rng.shuffle(positions)
    elif motif == "columns":
        positions.sort(key=lambda p: (p[1], p[0]))
    elif motif == "blocks":
        cr, cc = int(rng.integers(h)), int(rng.integers(w))
        positions.sort(key=lambda p: (abs(p[0] - cr) + abs(p[1] - cc), p))
    elif motif == "corner":
        positions.sort(key=lambda p: (p[0] != 0, p[1] != 0, p[0] + p[1], p))
    grid = np.empty(shape, dtype=np.int8)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color
    return grid


def _make_pair(rng, k, shape, used, palette=None, layout=None, style=None,
               forbidden_palettes=(), forbidden_layouts=()):
    if style is None:
        style = ("balanced", "skew", "random")[int(rng.choice(3, p=[.30, .25, .45]))]
    for attempt in range(80):
        colors = (tuple(int(v) for v in rng.choice(10, size=k, replace=False))
                  if palette is None or attempt >= 20 else palette)
        if tuple(sorted(colors)) in forbidden_palettes:
            continue
        if layout is not None and attempt < 20:
            grid = np.asarray(colors, dtype=np.int8)[np.asarray(layout[1]).reshape(shape)]
        else:
            motif = ("scatter", "rows", "columns", "blocks", "corner")[
                int(rng.choice(5, p=[.45, .16, .16, .14, .09]))]
            grid = _paint(rng, shape, colors, style, motif)
        signature = _key(grid)
        if signature in used:
            continue
        if shape == (3, 3) and signature[1] in _SOURCE_INPUTS:
            continue
        if _layout(grid) in forbidden_layouts:
            continue
        used.add(signature)
        return {"input": grid, "output": np.tile(grid, (k, k))}
    raise RuntimeError("could not construct a unique pair")


def _plan(rng, n_train, n_test):
    # A regime controls evidence and challenge, not the transformation.
    mode = ("source_like", "shape_shift", "smaller_return", "extrapolate", "mixed")[
        int(rng.integers(5))]
    if n_train < 3 and mode == "extrapolate":
        mode = "shape_shift"
    factors = [2, 3, 4] if n_train >= 3 else [2, 4]
    while len(factors) < n_train:
        if mode in ("source_like", "extrapolate", "shape_shift"):
            factors.append(int(rng.integers(2, 5)))
        else:
            factors.append(int(rng.choice([1, 2, 3, 4, 5, 6],
                                          p=[.08, .24, .24, .24, .12, .08])))
    if mode == "smaller_return":
        factors.sort()
    else:
        rng.shuffle(factors)
    if mode in ("source_like", "shape_shift"):
        shapes = [(3, 3)] * n_train
    else:
        shapes = [_choose_shape(rng, k) for k in factors]
        if n_train >= 3:
            shapes[1] = shapes[0] if shapes[0] in _shape_choices(factors[1]) else (3, 3)
            if len(set(shapes)) == 1:
                shapes[-1] = _choose_shape(rng, factors[-1], excluded={shapes[0]})
    tests = []
    for index in range(n_test):
        if mode == "source_like":
            k = int(rng.choice([f for f in factors if f >= 2]))
            shape = (3, 3)
        elif mode == "shape_shift":
            k = int(rng.choice([f for f in factors if 2 <= f <= 4]))
            shape = _choose_shape(rng, k, excluded={(3, 3)})
        elif mode == "smaller_return":
            k = int(rng.choice([2, 3])) if n_train >= 3 else 2
            matching = [shapes[i] for i, factor in enumerate(factors) if factor == k]
            shape = matching[int(rng.integers(len(matching)))] if index == 0 else _choose_shape(rng, k)
        elif mode == "extrapolate":
            k = 5 + index % 2
            shape = _choose_shape(rng, k)
        else:
            k = int(rng.choice([f for f in factors if f >= 2]))
            matching = [shapes[i] for i, factor in enumerate(factors) if factor == k]
            shape = matching[int(rng.integers(len(matching)))] if index == 0 else _choose_shape(rng, k)
        tests.append((k, shape))
    return mode, list(zip(factors, shapes)), tests


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
    n_train, n_test = _resolve_counts(seed, num_train, num_test)
    mode, train_plan, test_plan = _plan(rng, n_train, n_test)
    child_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                               size=n_train + n_test, dtype=np.uint64)
    used = set()
    train, test = [], []
    by_factor = {}
    for index, (k, shape) in enumerate(train_plan):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        style = "balanced" if k not in by_factor else "skew"
        pair = _make_pair(pair_rng, k, shape, used, style=style)
        train.append(pair)
        by_factor.setdefault(k, []).append(pair)
    for index, (k, shape) in enumerate(test_plan):
        pair_rng = np.random.default_rng(int(child_seeds[n_train + index]))
        palette, layout = None, None
        candidates = [p for p in by_factor.get(k, []) if p["input"].shape == shape]
        if mode in ("smaller_return", "mixed") and index == 0 and candidates:
            layout = _layout(candidates[int(pair_rng.integers(len(candidates)))]["input"])
        elif mode != "source_like" and candidates and pair_rng.random() < .22:
            layout = _layout(candidates[int(pair_rng.integers(len(candidates)))]["input"])
        elif mode != "source_like" and k in by_factor and pair_rng.random() < .35:
            exemplar = by_factor[k][int(pair_rng.integers(len(by_factor[k])))]
            palette = tuple(int(v) for v in np.unique(exemplar["input"]))
        old_palettes = {tuple(int(v) for v in np.unique(p["input"]))
                        for p in by_factor.get(k, [])} if mode == "source_like" else ()
        old_layouts = {_layout(p["input"]) for p in by_factor.get(k, [])}
        test.append(_make_pair(pair_rng, k, shape, used, palette=palette, layout=layout,
                               forbidden_palettes=old_palettes,
                               forbidden_layouts=old_layouts if mode == "source_like" else ()))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs, train_factors = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(all(1 <= side <= 30 for side in grid.shape), "illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "non-integer grid")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            inp, out = pair["input"], pair["output"]
            k = int(len(np.unique(inp)))
            _require(1 <= k <= 6, "unsupported color count")
            _require(out.shape == (k * inp.shape[0], k * inp.shape[1]),
                     "output dimensions do not follow color count")
            for row in range(k):
                for col in range(k):
                    block = out[row * inp.shape[0]:(row + 1) * inp.shape[0],
                                col * inp.shape[1]:(col + 1) * inp.shape[1]]
                    _require(np.array_equal(block, inp), "output tile differs from input")
            signature = _key(inp)
            pair_signature = signature, _key(out)
            _require(signature not in seen_inputs and pair_signature not in seen_pairs,
                     "duplicate generated pair")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            _require(not (inp.shape == (3, 3) and signature[1] in _SOURCE_INPUTS),
                     "source pair copied")
            if split == "train":
                train_factors.add(k)
            else:
                _require(k >= 2, "identity tests do not exercise tiling")
                if k not in train_factors:
                    _require({2, 3, 4}.issubset(train_factors) and k in {5, 6},
                             "unseen test factor lacks demonstrated progression")
    _require(2 in train_factors and (3 in train_factors or 4 in train_factors),
             "training lacks color-count contrast")
    if len(train) >= 3:
        _require({2, 3, 4}.issubset(train_factors),
                 "training must show the 2, 3, 4 progression")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzles = set(), set(), set()
    shapes, output_shapes, factors, layouts, palettes = set(), set(), set(), set(), set()
    frequency_contrast = shape_variety = smaller = novel = shape_shift = 0
    palette_reuse = layout_reuse = identities = 0
    same_k_new_shape = same_shape_new_k = 0
    rectangular = color_zero = axis_thirty = test_two_three = 0
    solid_rows = solid_columns = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_signature(puzzle))
        training, testing = puzzle["train"], puzzle["test"]
        train_ks = [len(np.unique(p["input"])) for p in training]
        train_shapes = {p["input"].shape for p in training}
        shape_variety += len(train_shapes) > 1
        pair_shapes = [p["input"].shape for p in training + testing]
        pair_ks = [len(np.unique(p["input"])) for p in training + testing]
        same_k_new_shape += any(len({pair_shapes[i] for i in range(len(pair_ks))
                                     if pair_ks[i] == k}) > 1 for k in set(pair_ks))
        same_shape_new_k += any(len({pair_ks[i] for i in range(len(pair_ks))
                                     if pair_shapes[i] == shape}) > 1
                                for shape in set(pair_shapes))
        counts_by_k = {}
        for pair in training + testing:
            inp, out = pair["input"], pair["output"]
            k = len(np.unique(inp))
            shapes.add(inp.shape)
            output_shapes.add(out.shape)
            factors.add(k)
            layouts.add(_layout(inp))
            palettes.add(tuple(int(v) for v in np.unique(inp)))
            identities += k == 1
            rectangular += inp.shape[0] != inp.shape[1]
            color_zero += bool(np.any(inp == 0))
            axis_thirty += max(out.shape) == 30
            solid_rows += any(len(set(row)) == 1 for row in inp.tolist())
            solid_columns += any(len(set(col)) == 1 for col in inp.T.tolist())
            counts = tuple(sorted(int(v) for v in np.unique(inp, return_counts=True)[1]))
            counts_by_k.setdefault(k, set()).add(counts)
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    _require(out[r, c] == inp[r % inp.shape[0], c % inp.shape[1]],
                             "self-test: incorrect periodic cell")
        frequency_contrast += any(len(v) >= 2 for v in counts_by_k.values())
        for pair in testing:
            inp = pair["input"]
            k = len(np.unique(inp))
            test_two_three += k in {2, 3}
            smaller += k < max(train_ks)
            novel += k not in train_ks
            shape_shift += inp.shape not in train_shapes and k in train_ks
            palette = tuple(int(v) for v in np.unique(inp))
            palette_reuse += any(tuple(int(v) for v in np.unique(p["input"])) == palette
                                 for p in training)
            layout_reuse += any(_layout(p["input"]) == _layout(inp) for p in training)
            # Infer the rule from copies of the demonstrations and the test input.
            for demo in training:
                demo_in = demo["input"].copy()
                demo_k = len(np.unique(demo_in))
                _require(np.array_equal(demo["output"].copy(),
                                        np.tile(demo_in, (demo_k, demo_k))),
                         "self-test: contradictory demonstration")
            _require(np.array_equal(np.tile(inp.copy(), (k, k)), pair["output"]),
                     "self-test: incorrect test prediction")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "insufficient count diversity")
    _require(len(puzzles) >= 40 and len(layouts) >= 100 and len(shapes) >= 12,
             "insufficient structural diversity")
    _require(len(output_shapes) >= 30 and len(palettes) >= 40,
             "insufficient output-size or palette diversity")
    _require({1, 2, 3, 4, 5, 6}.issubset(factors), "missing factor diversity")
    _require(frequency_contrast >= 10 and shape_variety >= 10,
             "insufficient within-puzzle frequency or dimension diversity")
    _require(same_k_new_shape >= 10 and same_shape_new_k >= 10,
             "tile shape and factor are too coupled")
    _require(rectangular >= 20 and color_zero >= 20 and axis_thirty >= 2,
             "missing rectangular, color-zero, or large-grid cases")
    _require(test_two_three >= 10 and solid_rows >= 20 and solid_columns >= 20,
             "missing small-factor tests or simple row/column motifs")
    _require(smaller >= 5 and novel >= 2 and shape_shift >= 5,
             "insufficient train/test regime diversity")
    _require(palette_reuse >= 1 and layout_reuse >= 1 and identities >= 1,
             "missing palette, layout, or identity cases")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(77, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count combination accepted")
    for args in ((314159, None, None), (912345, 7, 3)):
        _require(_signature(generate_puzzle(*args)) == _signature(generate_puzzle(*args)),
                 "whole-puzzle determinism failed")
