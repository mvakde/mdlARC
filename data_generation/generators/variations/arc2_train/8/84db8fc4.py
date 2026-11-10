import numpy as np


# Exact source inputs are excluded, without excluding their sizes or palettes.
_SOURCE_INPUTS = frozenset({
    "030300133101111300110310103030333003330011313000101010303303"
    "0000113010301331030011001331130011010000",
    "033031011313001131001010013033003313330100330000313300300030"
    "0033303033311130111300131003330330331311",
    "000000311300310110030100133131013000001001310103011003131010"
    "1003010300010110310303113003101133001303",
    "310333330310033013110111030003013303130013110130000113003113"
    "3010000030000331001333100110010130111113",
    "100101111300030301000103301333313110330111301301100301301103"
    "1130030333311113103130003310111031100010",
})
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _boundary(shape):
    h, w = shape
    result = np.zeros(shape, dtype=bool)
    result[0] = True
    result[-1] = True
    result[:, 0] = True
    result[:, -1] = True
    return result


def _components(mask):
    """Four-connected components of a Boolean grid."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cells = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in _NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                            and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(cells)
    return components


def _analyze(background):
    exterior = np.zeros(background.shape, dtype=bool)
    for component in _components(background):
        if any(r == 0 or c == 0 or r == background.shape[0] - 1
               or c == background.shape[1] - 1 for r, c in component):
            for r, c in component:
                exterior[r, c] = True
    enclosed = background & ~exterior
    return exterior, enclosed, _components(enclosed), _components(exterior)


def _diagonal_split(exterior, enclosed):
    return bool(np.any(
        (exterior[:-1, :-1] & enclosed[1:, 1:])
        | (enclosed[:-1, :-1] & exterior[1:, 1:])
        | (exterior[1:, :-1] & enclosed[:-1, 1:])
        | (enclosed[1:, :-1] & exterior[:-1, 1:])
    ))


def _shape(rng):
    # Independent side lengths allow squares, rectangles, and large grids.
    bands = ((8, 13), (12, 20), (19, 30))
    lo_h, hi_h = bands[int(rng.choice(3, p=[0.34, 0.39, 0.27]))]
    lo_w, hi_w = bands[int(rng.choice(3, p=[0.34, 0.39, 0.27]))]
    return int(rng.integers(lo_h, hi_h)), int(rng.integers(lo_w, hi_w))


def _gate_masks(rng):
    """Two chambers: one gate toggles a large region; another stays closed."""
    h = int(rng.integers(15, 30))
    w = int(rng.integers(15, 30))
    base = rng.random((h, w)) < rng.uniform(0.31, 0.51)
    ah = int(rng.integers(4, min(7, (h - 7) // 2 + 1)))
    aw = int(rng.integers(4, min(7, (w - 7) // 2 + 1)))
    bh = int(rng.integers(4, min(7, (h - 7) // 2 + 1)))
    bw = int(rng.integers(4, min(7, (w - 7) // 2 + 1)))
    ar = int(rng.integers(3, min(6, h // 4 + 2)))
    ac = int(rng.integers(3, min(6, w // 4 + 2)))
    br, bc = h - bh - 2, w - bw - 2

    for r, c, rh, cw in ((ar, ac, ah, aw), (br, bc, bh, bw)):
        base[r - 1:r + rh + 1, c - 1:c + cw + 1] = False
        base[r:r + rh, c:c + cw] = True
        # Roughen the chamber while retaining its large connected core.
        rough = rng.random((rh, cw)) < rng.uniform(0.04, 0.14)
        base[r:r + rh, c:c + cw][rough] = False
        cy, cx = r + rh // 2, c + cw // 2
        base[cy - 1:cy + 2, cx - 1:cx + 2] = True
        base[cy, cx] = False  # a preserved barrier island

    # An exterior path touches the closed chamber diagonally, never across its wall.
    base[br + bh, bc + bw:] = True
    base[br + bh - 1, bc + bw - 1] = True
    base[br + bh, bc + bw - 1] = False
    base[br + bh - 1, bc + bw] = False

    # Wall off a narrow, turning passage before carving it from the top edge.
    middle = ac + aw // 2
    base[:ar, ac - 1:ac + aw + 1] = False
    base[:2, middle - 2] = True
    base[1, middle - 2:middle + 2] = True
    base[1:ar - 1, middle + 1] = True
    base[ar - 2, middle:middle + 2] = True
    gate = (ar - 1, middle)
    base[ar:ar + ah // 2, middle] = True
    closed = base.copy()
    opened = base.copy()
    closed[gate] = False
    opened[gate] = True

    # Isolated perimeter regions make it explicit that all boundary components fill alike.
    for mask in (closed, opened):
        mask[0, 0] = True
        mask[0, 1] = False
        mask[1, 0] = False
        mask[-1, 0] = True
        mask[-1, 1] = False
        mask[-2, 0] = False
    return closed, opened


def _edge_background(rng):
    """Sparse boundary patches and only singleton enclosed components."""
    h, w = int(rng.integers(8, 16)), int(rng.integers(8, 16))
    mask = np.zeros((h, w), dtype=bool)
    start = int(rng.integers(1, w - 3))
    length = int(rng.integers(1, 5))
    mask[0, start:start + length] = True
    if rng.random() < 0.45:
        mask[1, start] = True
    positions = [(r, c) for r in range(2, h - 1, 3)
                 for c in range(1, w - 1, 3)]
    rng.shuffle(positions)
    for r, c in positions[:int(rng.integers(1, min(8, len(positions)) + 1))]:
        mask[r, c] = True
    return mask


def _random_background(rng, regime):
    for _ in range(96):
        shape = _shape(rng)
        barrier_probability = float(rng.uniform(0.39, 0.72))
        if regime == "target_test" and rng.random() < 0.3:
            barrier_probability = min(0.78, barrier_probability + 0.07)
        mask = rng.random(shape) >= barrier_probability
        exterior, enclosed, _, _ = _analyze(mask)
        if np.any(exterior) and np.any(enclosed):
            return mask
    # Deterministic bounded fallback for rare low-enclosure samples.
    mask = rng.random((12, 13)) > 0.58
    mask[0, 0] = True
    mask[5:8, 5:8] = False
    mask[6, 6] = True
    return mask


def _barrier_field(rng, background, palette, selected=None):
    if selected is None:
        count = int(rng.integers(1, len(palette) + 1))
        selected = rng.choice(palette, size=count, replace=False)
    selected = np.asarray(selected, dtype=np.int8)
    h, w = background.shape
    weights = rng.dirichlet(np.full(len(selected), 0.8))
    style = int(rng.integers(0, 3))
    if style == 0 or len(selected) == 1:
        field = rng.choice(selected, size=(h, w), p=weights).astype(np.int8)
    else:
        centers = rng.integers([0, 0], [h, w], size=(int(rng.integers(4, 10)), 2))
        center_colors = rng.choice(selected, size=len(centers), p=weights)
        rows, cols = np.indices((h, w))
        distances = ((rows[..., None] - centers[:, 0]) ** 2
                     + (cols[..., None] - centers[:, 1]) ** 2)
        field = center_colors[np.argmin(distances, axis=2)].astype(np.int8)
        if style == 2:
            mottled = rng.random((h, w)) < rng.uniform(0.12, 0.38)
            field[mottled] = rng.choice(selected, size=int(mottled.sum()), p=weights)

    # Every requested barrier color is actually visible, including in a gate pair.
    barriers = np.argwhere(~background)
    chosen = barriers[rng.choice(len(barriers), size=len(selected), replace=False)]
    for (r, c), color in zip(chosen, selected):
        field[r, c] = color
    return field


def _colorize(background, field, roles):
    bg, _, outside, inside = roles
    inp = np.where(background, bg, field).astype(np.int8)
    exterior, enclosed, _, _ = _analyze(background)
    out = inp.copy()
    out[exterior] = outside
    out[enclosed] = inside
    return {"input": inp, "output": out}


def _count_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = _count_argument("num_train", num_train, 2, 10)
    test = _count_argument("num_test", num_test, 1, 4)
    if train is None:
        train = int(rng.integers(max(2, (test or 0) + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _is_source(pair):
    inp = pair["input"]
    return inp.shape == (10, 10) and "".join(map(str, inp.flat)) in _SOURCE_INPUTS


def _new_pair(rng, regime, roles, seen):
    for _ in range(80):
        kind = int(rng.choice(3, p=[0.67, 0.15, 0.18] if regime == "source_train"
                              else [0.56, 0.17, 0.27]))
        if kind == 0:
            background = _random_background(rng, regime)
        elif kind == 1:
            background = _edge_background(rng)
        else:
            masks = _gate_masks(rng)
            background = masks[int(rng.integers(0, 2))]
        field = _barrier_field(rng, background, roles[1])
        pair = _colorize(background, field, roles)
        key = _key(pair["input"])
        if key not in seen and not _is_source(pair):
            seen.add(key)
            return pair
    raise RuntimeError("Could not construct a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    colors = rng.choice(10, size=7, replace=False)
    barrier_count = int(rng.integers(1, 5))
    roles = (int(colors[0]), tuple(int(x) for x in colors[1:1 + barrier_count]),
             int(colors[5]), int(colors[6]))

    seen = set()
    closed, opened = _gate_masks(rng)
    field = _barrier_field(rng, opened, roles[1], roles[1])
    gate_pairs = [_colorize(mask, field, roles) for mask in (closed, opened)]
    for pair in gate_pairs:
        seen.add(_key(pair["input"]))
    train = gate_pairs + [_new_pair(rng, "source_train", roles, seen)
                          for _ in range(train_count - 2)]
    test = [_new_pair(rng, "target_test", roles, seen)
            for _ in range(test_count)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles_from_training(pair):
    inp, out = pair["input"], pair["output"]
    changed = inp != out
    assert np.any(changed)
    changed_colors = np.unique(inp[changed])
    assert len(changed_colors) == 1
    bg = int(changed_colors[0])
    background = inp == bg
    assert np.array_equal(changed, background)
    on_boundary = background & _boundary(inp.shape)
    assert np.any(on_boundary)
    exterior_colors = np.unique(out[on_boundary])
    assert len(exterior_colors) == 1
    outside = int(exterior_colors[0])
    fill_colors = {int(x) for x in np.unique(out[background])}
    assert len(fill_colors) == 2 and outside in fill_colors
    inside = next(iter(fill_colors - {outside}))
    barriers = {int(x) for x in np.unique(inp[~background])}
    assert 1 <= len(barriers) <= 4
    assert len({bg, outside, inside} | barriers) == len(barriers) + 3
    return bg, barriers, outside, inside


def _validate_pair(pair, roles):
    bg, barriers, outside, inside = roles
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    assert bg in inp
    background = inp == bg
    assert {int(x) for x in np.unique(inp[~background])} <= barriers
    assert np.any(~background)
    exterior, enclosed, pockets, exterior_components = _analyze(background)
    assert np.any(exterior) and np.any(enclosed)
    expected = inp.copy()
    expected[exterior] = outside
    expected[enclosed] = inside
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {
        "pockets": len(pockets),
        "largest": max(map(len, pockets)),
        "exterior": int(exterior.sum()),
        "exterior_components": len(exterior_components),
        "diagonal": _diagonal_split(exterior, enclosed),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= side <= 30 for side in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        assert not _is_source(pair)
        input_key = _key(pair["input"])
        pair_key = (input_key, _key(pair["output"]))
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)

    roles = _roles_from_training(train[0])
    training_features = []
    for pair in train:
        inferred = _roles_from_training(pair)
        assert (inferred[0], inferred[2], inferred[3]) == (roles[0], roles[2], roles[3])
        assert inferred[1] <= roles[1]
        training_features.append(_validate_pair(pair, roles))
    for pair in test:
        _validate_pair(pair, roles)
    assert any(f["diagonal"] for f in training_features)
    assert any(f["largest"] >= 8 for f in training_features)
    assert any(f["exterior"] >= 8 for f in training_features)
    # A visible one-cell gate changes the fate of a whole region.
    a, b = train[0], train[1]
    assert a["input"].shape == b["input"].shape
    assert np.count_nonzero(a["input"] != b["input"]) == 1
    assert np.count_nonzero(a["output"] != b["output"]) >= 8


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for p in puzzle["train"] + puzzle["test"])


def _predict_from_examples(train, test_input):
    """Reconstruct the output using visible examples and input only."""
    example_in = train[0]["input"]
    example_out = train[0]["output"]
    changed = example_in != example_out
    background_color = int(np.unique(example_in[changed])[0])
    boundary = _boundary(example_in.shape)
    outside_color = int(np.unique(example_out[changed & boundary])[0])
    inside_color = next(int(x) for x in np.unique(example_out[changed])
                        if x != outside_color)
    background = test_input == background_color
    height, width = background.shape
    reached = np.zeros(background.shape, dtype=bool)
    stack = []
    for row in range(height):
        for column in range(width):
            if (row in (0, height - 1) or column in (0, width - 1)) \
                    and background[row, column]:
                reached[row, column] = True
                stack.append((row, column))
    while stack:
        row, column = stack.pop()
        for dr, dc in _NEIGHBORS:
            y, x = row + dr, column + dc
            if 0 <= y < height and 0 <= x < width and background[y, x] \
                    and not reached[y, x]:
                reached[y, x] = True
                stack.append((y, x))
    prediction = test_input.copy()
    prediction[background] = inside_color
    prediction[reached] = outside_color
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, role_signatures, layouts, fingerprints = set(), set(), set(), set()
    counts, sizes, barriers, exterior_sizes = set(), set(), set(), set()
    singletons = dense = multiexterior = short_exterior = islands = 0
    coherent = mottled = aspect = large_grid = 0
    train_pockets, test_pockets = set(), set()
    train_largest, test_largest = set(), set()
    test_denser = test_larger = 0
    varying_shape = varying_barrier_count = 0
    total = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        roles = _roles_from_training(puzzle["train"][0])
        role_signatures.add((roles[0], tuple(sorted(roles[1])), roles[2], roles[3]))
        local_shapes, local_barriers = set(), set()
        local_train_density, local_test_density = [], []
        for index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            inp = pair["input"]
            feature = _validate_pair(pair, roles)
            assert np.array_equal(
                _predict_from_examples(puzzle["train"], inp), pair["output"])
            total += 1
            shapes.add(inp.shape)
            local_shapes.add(inp.shape)
            layout = inp == roles[0]
            layouts.add((inp.shape, layout.tobytes()))
            counts.add(feature["pockets"])
            sizes.add(feature["largest"])
            exterior_sizes.add(feature["exterior"])
            nbar = len(np.unique(inp[~layout]))
            barriers.add(nbar)
            local_barriers.add(nbar)
            target_pockets = train_pockets if index < len(puzzle["train"]) else test_pockets
            target_largest = train_largest if index < len(puzzle["train"]) else test_largest
            target_pockets.add(feature["pockets"])
            target_largest.add(feature["largest"])
            singletons += feature["largest"] == 1
            fraction = np.count_nonzero(~layout) / layout.size
            dense += fraction > 0.65
            (local_train_density if index < len(puzzle["train"]) else local_test_density).append(fraction)
            multiexterior += feature["exterior_components"] >= 2
            short_exterior += feature["exterior"] <= 5
            large_grid += max(inp.shape) >= 25
            aspect += max(inp.shape) / min(inp.shape) >= 1.5
            interior = ~layout
            isolated = (interior[1:-1, 1:-1] & layout[:-2, 1:-1]
                        & layout[2:, 1:-1] & layout[1:-1, :-2]
                        & layout[1:-1, 2:])
            islands += bool(np.any(isolated))
            if nbar >= 2:
                adjacent = ((~layout[:, :-1]) & (~layout[:, 1:]),
                            (~layout[:-1, :]) & (~layout[1:, :]))
                total_adjacent = sum(int(x.sum()) for x in adjacent)
                same = (np.count_nonzero(adjacent[0] & (inp[:, :-1] == inp[:, 1:]))
                        + np.count_nonzero(adjacent[1] & (inp[:-1, :] == inp[1:, :])))
                if total_adjacent >= 20:
                    agreement = same / total_adjacent
                    coherent += agreement >= 0.75
                    mottled += agreement <= 0.55
        varying_shape += len(local_shapes) > 1
        varying_barrier_count += len(local_barriers) > 1
        test_denser += max(local_test_density) > float(np.median(local_train_density))
        test_larger += max(p["input"].size for p in puzzle["test"]) \
            > max(p["input"].size for p in puzzle["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(role_signatures) >= 30
    assert len(layouts) >= int(total * 0.95)
    assert len(fingerprints) == 42
    assert len(counts) >= 12 and len(sizes) >= 15
    assert len(exterior_sizes) >= 20 and barriers == {1, 2, 3, 4}
    assert singletons >= 5 and dense >= 15 and multiexterior >= 42
    assert varying_shape >= 35 and varying_barrier_count >= 12
    assert short_exterior >= 5 and islands >= 42
    assert coherent >= 10 and mottled >= 5
    assert aspect >= 20 and large_grid >= 40
    assert len(train_pockets & test_pockets) >= 8
    assert len(train_largest & test_largest) >= 8
    assert test_denser >= 10 and test_larger >= 3

    assert _fingerprint(puzzles[7]) == _fingerprint(generate_puzzle(7))
    a = generate_puzzle(987654, 7, 3)
    assert _fingerprint(a) == _fingerprint(generate_puzzle(987654, 7, 3))
    assert _fingerprint(generate_puzzle(91001, 6, 2)) != _fingerprint(generate_puzzle(91002, 6, 2))
    for seed, train, test in ((501, 2, 1), (502, 10, 1), (503, 10, 4),
                              (601, None, 1), (602, None, 4),
                              (603, 2, None), (604, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal count request accepted: {kwargs}")
