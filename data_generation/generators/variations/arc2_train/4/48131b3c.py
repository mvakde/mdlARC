"""Generate binary color-exchange and 2-by-2 tiling ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[7, 0], [0, 7]], dtype=np.int8),
    np.array([[0, 8, 0], [8, 0, 8], [0, 0, 0]], dtype=np.int8),
    np.array([[4, 0, 0, 0], [0, 4, 4, 4], [0, 0, 4, 0], [0, 4, 0, 0]], dtype=np.int8),
    np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.int8),
)
_FAMILIES = (
    "arbitrary", "sparse", "dense", "connected", "checkerboard",
    "stripes", "diagonal", "block", "clusters", "symmetric",
)


def _require(condition, message):
    if not bool(condition):
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


def _counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, bytes(np.asarray(grid, dtype=np.uint8).ravel())


def _source_copy(grid):
    return any(grid.shape == source.shape and np.array_equal(grid, source)
               for source in _SOURCE_INPUTS)


def _transform(grid):
    colors = np.unique(grid)
    tile = np.where(grid == colors[0], colors[1], colors[0])
    return np.tile(tile, (2, 2)).astype(np.int8)


def _has_nontrivial_symmetry(bits):
    """Check all seven nonidentity square rotations and reflections."""
    if bits.shape[0] != bits.shape[1]:
        return False
    transforms = [np.rot90(bits, k) for k in range(1, 4)]
    transforms.extend(np.fliplr(np.rot90(bits, k)) for k in range(4))
    return any(np.array_equal(bits, other) for other in transforms)


def _extent(rng, profile, minimum=2):
    if profile == "small":
        low, high = max(minimum, 2), 6
    elif profile == "large":
        low, high = max(minimum, 8), 15
    else:
        band = int(rng.integers(3))
        low, high = ((2, 4), (5, 9), (10, 15))[band]
        low = max(low, minimum)
    return int(rng.integers(low, high + 1))


def _shape(rng, profile):
    mode = int(rng.integers(10))
    if mode < 4:
        side = _extent(rng, profile)
        return side, side
    if mode == 9:
        length = _extent(rng, profile, 3)
        return (1, length) if int(rng.integers(2)) else (length, 1)
    a, b = _extent(rng, profile), _extent(rng, profile)
    if a == b:
        b = b + 1 if b < 15 else b - 1
    return (max(a, b), min(a, b)) if mode < 7 else (min(a, b), max(a, b))


def _palette(rng, available=None):
    pool = np.arange(10) if available is None else np.asarray(available)
    return tuple(sorted(int(x) for x in rng.choice(pool, 2, replace=False)))


def _palettes(rng, train_count, test_count):
    """Choose shared, mixed, or independently varying pair palettes."""
    total = train_count + test_count
    base = _palette(rng)
    mode = int(rng.choice(3, p=(0.22, 0.33, 0.45)))
    if mode == 0:
        return [base] * total
    result = [base]
    if mode == 2:
        result.append(_palette(rng, [color for color in range(10) if color not in base]))
    else:
        result.append(base if rng.random() < 0.5 else _palette(rng))
    for _ in range(total - 2):
        result.append(base if mode == 1 and rng.random() < 0.45 else _palette(rng))
    train = result[:train_count]
    rng.shuffle(train)
    return train + result[train_count:]


def _connected_mask(rng, height, width, target):
    start = (int(rng.integers(height)), int(rng.integers(width)))
    selected = {start}
    frontier = set()

    def add_neighbors(cell):
        row, col = cell
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and neighbor not in selected:
                frontier.add(neighbor)

    add_neighbors(start)
    while len(selected) < target:
        options = sorted(frontier)
        chosen = options[int(rng.integers(len(options)))]
        frontier.remove(chosen)
        selected.add(chosen)
        add_neighbors(chosen)
    bits = np.zeros((height, width), dtype=np.uint8)
    for row, col in selected:
        bits[row, col] = 1
    return bits


def _mask(rng, shape, family):
    height, width = shape
    total = height * width
    rows, cols = np.indices(shape)
    if family in ("arbitrary", "sparse", "dense"):
        if family == "sparse":
            count = int(rng.integers(1, max(2, total // 6 + 1)))
        elif family == "dense":
            count = total - int(rng.integers(1, max(2, total // 6 + 1)))
        else:
            count = min(total - 1, max(1, int(round(total * float(rng.uniform(0.12, 0.88))))))
        flat = np.zeros(total, dtype=np.uint8)
        flat[rng.choice(total, count, replace=False)] = 1
        bits = flat.reshape(shape)
    elif family == "connected":
        bits = _connected_mask(rng, height, width, int(rng.integers(1, total)))
    elif family == "checkerboard":
        scale_r = int(rng.integers(1, min(4, height + 1)))
        scale_c = int(rng.integers(1, min(4, width + 1)))
        phase_r = int(rng.integers(scale_r))
        phase_c = int(rng.integers(scale_c))
        bits = (((rows + phase_r) // scale_r + (cols + phase_c) // scale_c) % 2).astype(np.uint8)
    elif family == "stripes":
        use_rows = bool(rng.integers(2))
        axis = rows if use_rows else cols
        span = height if use_rows else width
        period = int(rng.integers(2, min(7, span + 2)))
        thickness = int(rng.integers(1, period))
        offset = int(rng.integers(period))
        bits = (((axis + offset) % period) < thickness).astype(np.uint8)
    elif family == "diagonal":
        slope = 1 if int(rng.integers(2)) else -1
        period = int(rng.integers(2, min(9, max(height, width) + 2)))
        offset = int(rng.integers(period))
        thickness = int(rng.integers(1, max(2, period // 2 + 1)))
        bits = (((rows + slope * cols + offset) % period) < thickness).astype(np.uint8)
    elif family == "block":
        block_h = int(rng.integers(1, height + 1))
        block_w = int(rng.integers(1, width + 1))
        top = int(rng.integers(height - block_h + 1))
        left = int(rng.integers(width - block_w + 1))
        bits = np.zeros(shape, dtype=np.uint8)
        bits[top:top + block_h, left:left + block_w] = 1
    elif family == "clusters":
        bits = np.zeros(shape, dtype=np.uint8)
        for _ in range(int(rng.integers(2, 6))):
            block_h = int(rng.integers(1, max(2, height // 2 + 1)))
            block_w = int(rng.integers(1, max(2, width // 2 + 1)))
            top = int(rng.integers(height - block_h + 1))
            left = int(rng.integers(width - block_w + 1))
            bits[top:top + block_h, left:left + block_w] = 1
    else:
        raw = (rng.random(shape) < float(rng.uniform(0.12, 0.5))).astype(np.uint8)
        choice = int(rng.integers(3))
        mirror = np.fliplr(raw) if choice == 0 else np.flipud(raw) if choice == 1 else np.rot90(raw, 2)
        bits = np.maximum(raw, mirror)
    if rng.random() < 0.5:
        bits = 1 - bits
    return bits.astype(np.uint8)


def _pair(rng, shape, palette, family, used, witness=False):
    for _ in range(256):
        bits = _mask(rng, shape, "arbitrary" if witness else family)
        if bits.min() == bits.max() or (witness and _has_nontrivial_symmetry(bits)):
            continue
        key = _grid_key(bits)
        if key in used:
            continue
        grid = np.where(bits, palette[1], palette[0]).astype(np.int8)
        if _source_copy(grid):
            continue
        used.add(key)
        return {"input": grid, "output": _transform(grid)}
    # Deterministic escape from a crowded small-mask space.
    for side in range(5, 16):
        for offset in range(side * side):
            bits = np.zeros((side, side), dtype=np.uint8)
            bits.flat[offset] = 1
            bits.flat[(offset + side + 2) % (side * side)] = 1
            if witness and _has_nontrivial_symmetry(bits):
                continue
            key = _grid_key(bits)
            if key not in used:
                grid = np.where(bits, palette[1], palette[0]).astype(np.int8)
                if not _source_copy(grid):
                    used.add(key)
                    return {"input": grid, "output": _transform(grid)}
    raise RuntimeError("could not construct a unique binary input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    palettes = _palettes(rng, train_count, test_count)
    shape_regime = int(rng.integers(4))
    train_profile = ("any", "small", "large", "any")[shape_regime]
    test_profile = ("any", "large", "small", "any")[shape_regime]
    witness_index = int(rng.integers(train_count))
    train_shapes = [_shape(rng, train_profile) for _ in range(train_count)]
    train_shapes[witness_index] = (int(rng.integers(4, 9)),) * 2
    test_shapes = []
    for _ in range(test_count):
        if shape_regime == 0 and rng.random() < 0.7:
            test_shapes.append(train_shapes[int(rng.integers(train_count))])
        else:
            test_shapes.append(_shape(rng, test_profile))
    used = set()
    puzzle = {"train": [], "test": []}
    for index, shape in enumerate(train_shapes + test_shapes):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
        family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
        pair = _pair(pair_rng, shape, palettes[index], family, used, witness=index == witness_index)
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen_inputs, seen_pairs, seen_masks = set(), set(), set()
    orientation_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integer colors")
                _require(np.all((grid >= 0) & (grid <= 9)), "illegal color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape[0] <= 15 and inp.shape[1] <= 15, "input too large to double")
            colors = np.unique(inp)
            _require(len(colors) == 2, "input must show both color roles")
            _require(out.shape == (2 * inp.shape[0], 2 * inp.shape[1]), "dimensions must double")
            tile = np.where(inp == colors[0], colors[1], colors[0])
            expected = np.concatenate((np.concatenate((tile, tile), axis=1),
                                       np.concatenate((tile, tile), axis=1)), axis=0)
            _require(np.array_equal(out, expected), "wrong color exchange or tiling")
            _require(not _source_copy(inp), "source pair copied")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            bits = (inp == colors[1]).astype(np.uint8)
            mask_key = _grid_key(bits)
            _require(mask_key not in seen_masks, "binary layout reused")
            seen_masks.add(mask_key)
            if split == "train" and inp.shape[0] == inp.shape[1] and not _has_nontrivial_symmetry(bits):
                orientation_witness = True
    _require(orientation_witness, "training needs an asymmetric square orientation witness")


def _puzzles_equal(left, right):
    return all(len(left[split]) == len(right[split]) for split in ("train", "test")) and all(
        np.array_equal(a[name], b[name]) for split in ("train", "test")
        for a, b in zip(left[split], right[split]) for name in ("input", "output"))


def self_test() -> None:
    train_counts, test_counts, shapes, palettes, layouts = set(), set(), set(), set(), set()
    nonzero, zero, rectangular, tall, wide, line, large, sparse, dense = (0,) * 9
    shared_palette_puzzles = variable_palette_puzzles = 0
    split_features = {"train": {"sparse": 0, "dense": 0, "checker": 0, "stripe": 0,
                                "symmetric": 0},
                      "test": {"sparse": 0, "dense": 0, "checker": 0, "stripe": 0,
                               "symmetric": 0}}
    shared_shapes = familiar_test_palettes = new_test_palettes = 0
    larger_tests = smaller_tests = mixed_zero_puzzles = combined_axes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes = []
        train_shapes = []
        test_shapes = []
        train_palettes = []
        test_palettes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                colors = np.unique(inp)
                pair_palettes.append(tuple(int(x) for x in colors))
                swapped = np.empty_like(inp)
                swapped[inp == colors[0]] = colors[1]
                swapped[inp == colors[1]] = colors[0]
                h, w = inp.shape
                for r in (0, h):
                    for c in (0, w):
                        _require(np.array_equal(out[r:r+h, c:c+w], swapped), "self-test transformation failed")
                shapes.add(inp.shape)
                palettes.add(tuple(int(x) for x in colors))
                layouts.add(_grid_key((inp == colors[1]).astype(np.uint8)))
                _require(not _source_copy(inp), "self-test found an exact source input")
                if split == "train":
                    train_shapes.append(inp.shape)
                    train_palettes.append(tuple(int(x) for x in colors))
                else:
                    test_shapes.append(inp.shape)
                    test_palettes.append(tuple(int(x) for x in colors))
                nonzero += int(0 not in colors)
                zero += int(0 in colors)
                rectangular += int(h != w)
                tall += int(h > w)
                wide += int(w > h)
                line += int(min(h, w) == 1)
                large += int(max(h, w) >= 10)
                count = int(np.count_nonzero(inp == colors[1]))
                sparse += int(count <= max(1, h*w // 10))
                dense += int(count >= h*w - max(1, h*w // 10))
                split_features[split]["sparse"] += int(count <= max(1, h*w // 10))
                split_features[split]["dense"] += int(count >= h*w - max(1, h*w // 10))
                checker = (np.indices(inp.shape).sum(axis=0) % 2).astype(np.uint8)
                bits = (inp == colors[1]).astype(np.uint8)
                split_features[split]["checker"] += int(np.array_equal(bits, checker) or
                                                         np.array_equal(bits, 1 - checker))
                split_features[split]["stripe"] += int(all(np.all(row == row[0]) for row in bits) or
                                                        all(np.all(bits[:, col] == bits[0, col]) for col in range(w)))
                split_features[split]["symmetric"] += int(h == w and _has_nontrivial_symmetry(bits))
                combined_axes += int(h != w and max(h, w) >= 10 and 0 not in colors and
                                     (count <= max(1, h*w // 10) or count >= h*w - max(1, h*w // 10)))
        shared_palette_puzzles += int(len(set(pair_palettes)) == 1)
        variable_palette_puzzles += int(any(set(a).isdisjoint(b) for a in pair_palettes for b in pair_palettes))
        mixed_zero_puzzles += int(any(0 in palette for palette in pair_palettes) and
                                  any(0 not in palette for palette in pair_palettes))
        shared_shapes += int(bool(set(train_shapes) & set(test_shapes)))
        familiar_test_palettes += int(bool(set(train_palettes) & set(test_palettes)))
        new_test_palettes += int(bool(set(test_palettes) - set(train_palettes)))
        train_areas = [h * w for h, w in train_shapes]
        test_areas = [h * w for h, w in test_shapes]
        larger_tests += int(max(test_areas) > max(train_areas))
        smaller_tests += int(min(test_areas) < min(train_areas))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(len(shapes) >= 30 and len(layouts) >= 140 and len(palettes) >= 20, "structure lacks diversity")
    _require(min(nonzero, zero, rectangular, tall, wide, line, large, sparse, dense) > 0,
             "a requested shape, palette, or density class is missing")
    _require(shared_palette_puzzles > 0 and variable_palette_puzzles > 0, "palette regimes are missing")
    _require(min(shared_shapes, familiar_test_palettes, new_test_palettes,
                 larger_tests, smaller_tests, mixed_zero_puzzles, combined_axes) > 0,
             "a train/test regime or compatible axis combination is missing")
    _require(all(min(features.values()) > 0 for features in split_features.values()),
             "train and test must each include sparse, dense, regular, and symmetric masks")
    for seed, train_count, test_count in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                                          (103, None, 1), (104, None, 4),
                                          (105, 2, None), (106, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
        _require(_puzzles_equal(puzzle, generate_puzzle(seed, train_count, test_count)), "nondeterministic")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    regression = generate_puzzle(142)
    anti_diagonal_only = np.array([[0, 0, 0, 1], [0, 0, 0, 0],
                                   [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    _require(_has_nontrivial_symmetry(anti_diagonal_only),
             "anti-diagonal reflection was not detected")
    _require(any(p["input"].shape[0] == p["input"].shape[1] and
                 not _has_nontrivial_symmetry((p["input"] == np.unique(p["input"])[1]).astype(np.uint8))
                 for p in regression["train"]), "seed-142 orientation regression")
    broken = generate_puzzle(999, 2, 1)
    wrong_output = broken["test"][0]["output"].copy()
    wrong_output[0, 0] = (int(wrong_output[0, 0]) + 1) % 10
    broken["test"][0]["output"] = wrong_output
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    ambiguous = np.where(anti_diagonal_only, 2, 0).astype(np.int8)
    symmetric = np.array([[0, 3], [3, 0]], dtype=np.int8)
    test_input = np.array([[4, 0, 4], [0, 4, 0], [4, 4, 0]], dtype=np.int8)
    ambiguous_puzzle = {
        "train": [{"input": a, "output": _transform(a)} for a in (ambiguous, symmetric)],
        "test": [{"input": test_input, "output": _transform(test_input)}],
    }
    try:
        validate_generated_puzzle(ambiguous_puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted orientation-ambiguous training")
    _require(not _puzzles_equal(generate_puzzle(700, 6, 2), generate_puzzle(701, 6, 2)),
             "different seeds produced identical puzzles")
