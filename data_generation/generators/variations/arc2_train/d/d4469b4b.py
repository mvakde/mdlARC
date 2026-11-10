"""Generate color-to-symbol ARC puzzles with irrelevant input geometry."""

import numpy as np


_SOURCE_ROWS = (
    ("10100", "10011", "11010", "01010", "10001"),
    ("20202", "20002", "22000", "20022", "22202"),
    ("03030", "33000", "03000", "00300", "33300"),
    ("00000", "00111", "01011", "01010", "00001"),
    ("30000", "00033", "03300", "03030", "30330"),
    ("02020", "02220", "02202", "22200", "00202"),
    ("20000", "02002", "20020", "00022", "00220"),
    ("03033", "00300", "30000", "00303", "00003"),
    ("11110", "00101", "01000", "01001", "00100"),
)
_SOURCE_KEYS = frozenset((5, 5, tuple(int(c) for row in rows for c in row)) for rows in _SOURCE_ROWS)

# The source codewords are cross, T, and a bottom-bar/right-column V pentomino.
_SHAPES = (
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),  # cross
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),  # T
    ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)),  # V
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),  # U
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),  # W
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),  # P
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),  # bent stroke
)
_DIMS = ((4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5),
         (4, 7), (7, 4), (6, 6), (5, 8), (8, 5), (6, 9), (9, 6))
_MODES = ("scatter", "blob", "stroke", "border", "hole", "singletons",
          "symmetric", "uneven", "decoy")


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _integer("num_train", num_train, 2, 10)
    test = None if num_test is None else _integer("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = seed % 4 + 1
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _normalize(points):
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((r - min_r, c - min_c) for r, c in points))


def _oriented(shape, turns, mirror):
    points = list(shape)
    if mirror:
        points = [(r, -c) for r, c in points]
    for _ in range(turns):
        points = [(c, -r) for r, c in points]
    return _normalize(points)


def _symbol(rng, shape, canvas):
    height, width = canvas
    for _ in range(24):
        points = _oriented(shape, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
        box_h = max(r for r, _ in points) + 1
        box_w = max(c for _, c in points) + 1
        if box_h <= height and box_w <= width:
            top = int(rng.integers(0, height - box_h + 1))
            left = int(rng.integers(0, width - box_w + 1))
            mask = np.zeros(canvas, dtype=bool)
            for r, c in points:
                mask[top + r, left + c] = True
            return mask
    raise AssertionError("shape cannot fit output canvas")


def _codebook(rng, colors):
    canvas = ((3, 3), (3, 4), (4, 3), (4, 4))[int(rng.choice(4, p=(.52, .16, .16, .16)))]
    source_like = canvas == (3, 3) and rng.random() < .35
    if source_like:
        masks = []
        for shape in _SHAPES[:3]:
            mask = np.zeros(canvas, dtype=bool)
            for r, c in shape:
                mask[r, c] = True
            masks.append(mask)
    else:
        order = rng.permutation(len(_SHAPES))
        masks = []
        seen = set()
        for index in order:
            for _ in range(16):
                mask = _symbol(rng, _SHAPES[int(index)], canvas)
                key = tuple(mask.flat)
                if key not in seen:
                    seen.add(key)
                    masks.append(mask)
                    break
            if len(masks) == 3:
                break
        assert len(masks) == 3
    rng.shuffle(masks)
    return {color: mask for color, mask in zip(colors, masks)}, canvas


def _roles(rng):
    background = int(rng.choice((0, 0, 0, 4, 6, 8, 9)))
    choices = [c for c in range(1, 10) if c != background]
    ink = int(rng.choice(choices))
    choices.remove(ink)
    colors = tuple(int(c) for c in rng.choice(choices, size=3, replace=False))
    return background, ink, colors


def _labels(rng, colors, train_count, test_count):
    if train_count == 2:
        trained = [int(c) for c in rng.choice(colors, size=2, replace=False)]
    else:
        trained = list(colors)
        while len(trained) < train_count:
            trained.append(int(rng.choice(colors)))
    rng.shuffle(trained)
    if test_count == 1 or rng.random() < .35:
        tested = [int(rng.choice(trained))] * test_count
    else:
        distinct = list(dict.fromkeys(trained))
        rng.shuffle(distinct)
        tested = distinct[:min(len(distinct), test_count)]
        while len(tested) < test_count:
            tested.append(int(rng.choice(distinct)))
        rng.shuffle(tested)
    return trained, tested


def _input_dimensions(rng, train_count, test_count):
    if rng.random() < .32:
        return [(5, 5)] * (train_count + test_count)
    train_dims = [_DIMS[int(i)] for i in rng.integers(0, len(_DIMS), size=train_count)]
    if len(set(train_dims)) == 1:
        train_dims[1] = _DIMS[(_DIMS.index(train_dims[0]) + 1) % len(_DIMS)]
    test_dims = []
    for _ in range(test_count):
        pool = [d for d in _DIMS if d not in train_dims] if rng.random() < .65 else list(_DIMS)
        test_dims.append(pool[int(rng.integers(0, len(pool)))])
    return train_dims + test_dims


def _grow(rng, height, width, count):
    mask = np.zeros((height, width), dtype=bool)
    start = int(rng.integers(0, height * width))
    mask.flat[start] = True
    while int(mask.sum()) < count:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(0, len(choices)))]
        mask[r, c] = True
    return mask


def _input_mask(rng, dims, mode, other_symbol):
    height, width = dims
    count = int(rng.integers(max(3, int(.14 * height * width)),
                             min(height * width - 3, int(.64 * height * width)) + 1))
    mask = np.zeros(dims, dtype=bool)
    if mode == "scatter":
        mask.flat[rng.choice(height * width, count, replace=False)] = True
    elif mode == "blob":
        mask = _grow(rng, height, width, count)
    elif mode == "stroke":
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            row = int(rng.integers(0, height))
            mask[row, :] = True
        else:
            col = int(rng.integers(0, width))
            mask[:, col] = True
        while int(mask.sum()) < count:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            choices = sorted(frontier)
            nr, nc = choices[int(rng.integers(0, len(choices)))]
            mask[nr, nc] = True
    elif mode == "border":
        top = int(rng.integers(0, height - 2))
        left = int(rng.integers(0, width - 2))
        bottom = int(rng.integers(top + 2, height))
        right = int(rng.integers(left + 2, width))
        mask[top, left:right + 1] = True
        mask[bottom, left:right + 1] = True
        mask[top:bottom + 1, left] = True
        mask[top:bottom + 1, right] = True
    elif mode == "hole":
        top = int(rng.integers(0, height - 2))
        left = int(rng.integers(0, width - 2))
        bottom = int(rng.integers(top + 2, height))
        right = int(rng.integers(left + 2, width))
        mask[top:bottom + 1, left:right + 1] = True
        mask[top + 1:bottom, left + 1:right] = False
        if int(mask.sum()) > int(.75 * height * width):
            mask[:] = False
            mask[top:top + 3, left:left + 3] = True
            mask[top + 1, left + 1] = False
    elif mode == "singletons":
        candidates = [tuple(p) for p in rng.permutation(np.indices(dims).reshape(2, -1).T)]
        for r, c in candidates:
            if any(mask[nr, nc] for nr, nc in ((r + dr, c + dc) for dr, dc in
                   ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < height and 0 <= c + dc < width)):
                continue
            mask[r, c] = True
            if int(mask.sum()) >= count:
                break
    elif mode == "symmetric":
        horizontal = bool(rng.integers(0, 2))
        candidates = [(r, c) for r in range(height) for c in range(width)
                      if (r <= (height - 1) // 2 if horizontal else c <= (width - 1) // 2)]
        rng.shuffle(candidates)
        for r, c in candidates:
            mask[r, c] = True
            mask[height - 1 - r if horizontal else r,
                 c if horizontal else width - 1 - c] = True
            if int(mask.sum()) >= count:
                break
    elif mode == "uneven":
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        rr, cc = np.indices(dims)
        weights = (1 + 4 * (rr == row) + 4 * (cc == col)).astype(float).flat
        weights = np.fromiter(weights, float)
        weights /= weights.sum()
        mask.flat[rng.choice(height * width, count, replace=False, p=weights)] = True
    elif mode == "decoy":
        coords = np.argwhere(other_symbol)
        box_h = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        box_w = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        top = int(rng.integers(0, height - box_h + 1))
        left = int(rng.integers(0, width - box_w + 1))
        for r, c in coords:
            mask[top + int(r - coords[:, 0].min()), left + int(c - coords[:, 1].min())] = True
    else:
        raise AssertionError("unknown geometry mode")
    # Keep several background cells visible even when the input color is nonzero.
    maximum = min(height * width - 3, int(.75 * height * width))
    if int(mask.sum()) > maximum:
        removable = [tuple(p) for p in np.argwhere(mask)]
        rng.shuffle(removable)
        for r, c in removable[:int(mask.sum()) - maximum]:
            mask[r, c] = False
    if int(mask.sum()) < 2:
        available = np.flatnonzero(~mask)
        mask.flat[rng.choice(available, 2 - int(mask.sum()), replace=False)] = True
    return mask


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _make_input(rng, dims, mode, color, background, other_symbol, seen, contrast_counts=()):
    for _ in range(96):
        use_mode = mode if _ < 32 else "scatter"
        mask = _input_mask(rng, dims, use_mode, other_symbol)
        if contrast_counts and max(abs(int(mask.sum()) - count) for count in contrast_counts) < 3:
            continue
        grid = np.where(mask, color, background).astype(np.int8)
        key = _grid_key(grid)
        if key not in seen and (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat)) not in _SOURCE_KEYS:
            seen.add(key)
            return grid
    raise RuntimeError("could not make a unique input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    background, ink, colors = _roles(rng)
    book, canvas = _codebook(rng, colors)
    train_labels, test_labels = _labels(rng, colors, train_count, test_count)
    dimensions = _input_dimensions(rng, train_count, test_count)
    mode_order = list(_MODES)
    rng.shuffle(mode_order)
    modes = [mode_order[i % len(mode_order)] for i in range(train_count + test_count)]
    # Give tests a new geometry mode for their label when possible.
    for i, label in enumerate(test_labels):
        used = {modes[j] for j, c in enumerate(train_labels) if c == label}
        available = [m for m in _MODES if m not in used]
        modes[train_count + i] = available[int(rng.integers(0, len(available)))] if available else modes[train_count + i]

    # One moderate-density binary silhouette can deliberately carry two labels.
    was_mixed = len(set(dimensions[:train_count])) > 1
    match = None
    if train_count >= 3 and rng.random() < .48:
        first_indices = [train_labels.index(color) for color in dict.fromkeys(train_labels)]
        candidates = [(i, j) for i in first_indices for j in first_indices
                      if i < j and train_labels[i] != train_labels[j]]
        match = candidates[int(rng.integers(0, len(candidates)))]
        dimensions[match[1]] = dimensions[match[0]]
        if was_mixed and len(set(dimensions[:train_count])) == 1:
            for k in range(train_count):
                if k not in match:
                    dimensions[k] = (6, 5) if dimensions[0] == (5, 5) else (5, 5)
                    break

    seen = set()
    pairs = []
    matched_mask = None
    counts_by_label = {color: [] for color in colors}
    for i, label in enumerate(train_labels + test_labels):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        dims = dimensions[i]
        if match is not None and i == match[1]:
            mask = matched_mask
            input_grid = np.where(mask, label, background).astype(np.int8)
            assert _grid_key(input_grid) not in seen
            seen.add(_grid_key(input_grid))
        elif match is not None and i == match[0]:
            for _ in range(96):
                mask = _input_mask(pair_rng, dims, "scatter", book[label])
                count = int(mask.sum())
                if .23 <= count / mask.size <= .48:
                    input_grid = np.where(mask, label, background).astype(np.int8)
                    other_grid = np.where(mask, train_labels[match[1]], background).astype(np.int8)
                    if (_grid_key(input_grid) not in seen and
                            (dims[0], dims[1], tuple(int(x) for x in input_grid.flat)) not in _SOURCE_KEYS and
                            (dims[0], dims[1], tuple(int(x) for x in other_grid.flat)) not in _SOURCE_KEYS):
                        break
            else:
                raise RuntimeError("could not make matched mask")
            matched_mask = mask
            seen.add(_grid_key(input_grid))
        else:
            other = [c for c in colors if c != label]
            decoy_color = int(pair_rng.choice(other))
            prior = counts_by_label[label] if i < train_count else []
            contrast = prior if prior and max(prior) - min(prior) < 3 else []
            input_grid = _make_input(pair_rng, dims, modes[i], label, background,
                                     book[decoy_color], seen, contrast)
        output_grid = np.where(book[label], ink, background).astype(np.int8)
        pairs.append({"input": input_grid, "output": output_grid})
        if i < train_count:
            counts_by_label[label].append(int(np.count_nonzero(input_grid == label)))
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def _connected(mask):
    points = {tuple(int(x) for x in p) for p in np.argwhere(mask)}
    reached = {next(iter(points))}
    frontier = list(reached)
    while frontier:
        r, c = frontier.pop()
        for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if p in points and p not in reached:
                reached.add(p)
                frontier.append(p)
    return len(reached) == len(points)


def _crop_mask(mask):
    points = np.argwhere(mask)
    return mask[points[:, 0].min():points[:, 0].max() + 1,
                points[:, 1].min():points[:, 1].max() + 1]


def _has_hole(mask):
    empty = {tuple(int(x) for x in p) for p in np.argwhere(~mask)}
    edge = [p for p in empty if p[0] in (0, mask.shape[0] - 1)
            or p[1] in (0, mask.shape[1] - 1)]
    reached = set(edge)
    while edge:
        r, c = edge.pop()
        for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if p in empty and p not in reached:
                reached.add(p)
                edge.append(p)
    return len(reached) < len(empty)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    pairs = train + test
    input_sets = []
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        _check_grid(pair["input"])
        _check_grid(pair["output"])
        assert pair["input"].shape[0] >= 4 and pair["input"].shape[1] >= 4
        input_sets.append(set(int(x) for x in np.unique(pair["input"])))
        assert len(input_sets[-1]) == 2
    shared = set.intersection(*input_sets)
    assert len(shared) == 1
    background = next(iter(shared))
    output_shapes = {pair["output"].shape for pair in pairs}
    assert len(output_shapes) == 1
    output_shape = next(iter(output_shapes))
    assert output_shape in {(3, 3), (3, 4), (4, 3), (4, 4)}
    seen_inputs = set()
    outputs_by_label = {}
    ink = None
    labels = []
    for pair, colors in zip(pairs, input_sets):
        input_grid, output_grid = pair["input"], pair["output"]
        assert background in colors and background in set(int(x) for x in np.unique(output_grid))
        label = next(iter(colors - {background}))
        labels.append(label)
        assert label != 0 and label != background
        assert 2 <= int(np.count_nonzero(input_grid == label)) <= min(input_grid.size - 3, int(.75 * input_grid.size))
        output_colors = set(int(x) for x in np.unique(output_grid))
        assert len(output_colors) == 2 and background in output_colors
        pair_ink = next(iter(output_colors - {background}))
        assert pair_ink != label and pair_ink != 0
        if ink is None:
            ink = pair_ink
        assert pair_ink == ink
        symbol = output_grid == ink
        assert int(symbol.sum()) == 5 and _connected(symbol)
        if label in outputs_by_label:
            assert np.array_equal(outputs_by_label[label], output_grid)
        else:
            outputs_by_label[label] = output_grid
        key = _grid_key(input_grid)
        assert key not in seen_inputs
        assert (input_grid.shape[0], input_grid.shape[1], tuple(int(x) for x in input_grid.flat)) not in _SOURCE_KEYS
        seen_inputs.add(key)
    assert len(set(_grid_key(g) for g in outputs_by_label.values())) == len(outputs_by_label)
    train_labels, test_labels = labels[:len(train)], labels[len(train):]
    assert len(set(train_labels)) == min(3, len(train))
    assert set(test_labels).issubset(set(train_labels))
    for label in set(train_labels):
        same_label = [pair["input"] != background for pair, c in zip(train, train_labels) if c == label]
        assert len({(mask.shape, tuple(mask.flat)) for mask in same_label}) == len(same_label)
    # Direct prediction from returned demonstrations, without hidden metadata.
    demonstrated = {c: out for c, out in zip(train_labels, (p["output"] for p in train))}
    for pair, label in zip(test, test_labels):
        assert np.array_equal(pair["output"], demonstrated[label])


def _same_puzzle(a, b):
    return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output")) and all(
                   len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(314159, 7, 3)
    assert _same_puzzle(a, generate_puzzle(314159, 7, 3))
    assert not _same_puzzle(a, generate_puzzle(314160, 7, 3))
    for seed, ntrain, ntest in ((700, 2, 1), (701, 10, 1), (702, 10, 4),
                                (703, None, 1), (704, None, 4),
                                (705, 2, None), (706, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(900, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts = set(), set()
    palettes, canvases, dimensions, codebooks, layouts = set(), set(), set(), set(), set()
    symbol_masks = set()
    queried = set()
    mixed_sizes = matched = contrasts = source_like = 0
    nonzero_background = non3_canvas = new_test_size = combined = 0
    single_tests = repeated_tests = mixed_tests = 0
    connected = disconnected = isolated = holes = borders = decoys = 0
    original_symbols = {
        tuple((np.array([[int((r, c) in shape) for c in range(3)]
                         for r in range(3)], dtype=bool)).flat)
        for shape in _SHAPES[:3]
    }
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        background = next(iter(set.intersection(*(
            set(int(x) for x in np.unique(p["input"])) for p in train))))
        ink = next(iter(set(int(x) for x in np.unique(train[0]["output"])) - {background}))
        nonzero_background += background != 0
        non3_canvas += train[0]["output"].shape != (3, 3)
        new_test_size += any(p["input"].shape not in {q["input"].shape for q in train}
                             for p in test)
        train_counts.add(len(train))
        test_counts.add(len(test))
        train_dims = {p["input"].shape for p in train}
        mixed_sizes += len(train_dims) > 1
        combined += background != 0 and len(train_dims) > 1 and train[0]["output"].shape != (3, 3)
        masks_by_label = {}
        counts_by_label = {}
        demonstrated = {}
        for pair in train:
            inp, out = pair["input"], pair["output"]
            label = next(iter(set(int(x) for x in np.unique(inp)) - {background}))
            masks_by_label.setdefault(label, set()).add((inp.shape, tuple((inp != background).flat)))
            counts_by_label.setdefault(label, []).append(int(np.count_nonzero(inp == label)))
            demonstrated[label] = out
        repeated = [counts for counts in counts_by_label.values() if len(counts) > 1]
        if repeated:
            assert any(max(counts) - min(counts) >= 3 for counts in repeated)
            contrasts += 1
        test_labels = [next(iter(set(int(x) for x in np.unique(p["input"])) - {background}))
                       for p in test]
        single_tests += len(set(test_labels)) == 1
        repeated_tests += len(test_labels) > 1 and len(set(test_labels)) < len(test_labels)
        mixed_tests += len(set(test_labels)) > 1
        if len(demonstrated) == 3 and train[0]["output"].shape == (3, 3):
            source_like += {tuple((out == ink).flat) for out in demonstrated.values()} == original_symbols
        masks = [(label, mask) for label, group in masks_by_label.items() for mask in group]
        matched += any(a != b and ma == mb and 4 <= sum(ma[1]) <= 18
                       for i, (a, ma) in enumerate(masks) for b, mb in masks[i + 1:])
        observed_masks = []
        for pair_index, pair in enumerate(train + test):
            inp, out = pair["input"], pair["output"]
            colors = set(int(x) for x in np.unique(inp))
            label = next(iter(colors - {background}))
            palettes.add((background, ink))
            canvases.add(out.shape)
            dimensions.add(inp.shape)
            mask = inp != background
            observed_masks.append((label, mask))
            layouts.add((inp.shape, tuple(mask.flat)))
            symbol_masks.add((out.shape, tuple((out == ink).flat)))
            if pair_index >= len(train):
                queried.add(label)
            assert int(np.count_nonzero(out == ink)) == 5
            connected += _connected(mask)
            disconnected += not _connected(mask)
            points = {tuple(int(x) for x in p) for p in np.argwhere(mask)}
            isolated += len(points) >= 3 and all(
                (r + dr, c + dc) not in points for r, c in points
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            holes += _has_hole(mask)
            crop = _crop_mask(mask)
            borders += (crop.shape[0] >= 3 and crop.shape[1] >= 3 and
                        bool(np.all(crop[0])) and bool(np.all(crop[-1])) and
                        bool(np.all(crop[:, 0])) and bool(np.all(crop[:, -1])) and
                        bool(np.any(~crop[1:-1, 1:-1])))
            if pair_index >= len(train):
                assert np.array_equal(out, demonstrated[label])
        for label, mask in observed_masks:
            if int(mask.sum()) == 5 and any(
                    other != label and np.array_equal(_crop_mask(mask), _crop_mask(out == ink))
                    for other, out in demonstrated.items()):
                decoys += 1
        codebooks.add(tuple(sorted((label, _grid_key(out)) for label, out in demonstrated.items())))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 12 and len(canvases) >= 3 and len(dimensions) >= 8
    assert len(codebooks) >= 40 and len(layouts) >= 100 and len(symbol_masks) >= 40
    assert mixed_sizes >= 12 and matched >= 4 and len(queried) >= 5
    assert contrasts >= 20 and source_like >= 3
    assert nonzero_background >= 15 and non3_canvas >= 10 and new_test_size >= 12
    assert single_tests >= 8 and repeated_tests >= 5 and mixed_tests >= 8
    assert connected >= 60 and disconnected >= 60 and isolated >= 10
    assert holes >= 10 and borders >= 10 and decoys >= 10 and combined >= 5
