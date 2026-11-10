import numpy as np


# Source answers are regression fixtures only; generation never crops them.
_SOURCE_OUTPUT_ROWS = {
    1: (
        "080888888888",
        "080800000008",
        "080808888808",
        "080808000808",
        "080880880808",
        "080008080808",
        "088888080808",
        "000000080808",
        "888888880808",
        "000000000808",
        "888888888808",
        "000000000008",
    ),
    2: (
        "8000000000000000",
        "8088888888888880",
        "8080000000000080",
        "8080888888888080",
        "8080800000008080",
        "8080808888808080",
        "8080808000808080",
        "8080880880808080",
        "8080008080808080",
        "8088888080808080",
        "8000000080808080",
        "8888888880808080",
        "0000000000808080",
        "8888888888808080",
        "0000000000008080",
        "8888888888888080",
    ),
    3: (
        "0808888888888888880000",
        "8808000000000000088880",
        "8008088888888888000080",
        "8088080000000008888080",
        "8080080888888800008080",
        "8080880800000888808080",
        "8080800808880000808080",
        "8080808808088880808080",
        "8080808008000080808080",
        "8080808880888080808080",
        "8080800008008080808080",
        "8080888808088080808080",
        "8080000888080080808080",
        "8088880000080880808080",
        "8000088888880800808080",
        "8888000000000808808080",
        "0008888888888808008080",
        "8800000000000008088080",
        "0888888888888888080080",
        "0000000000000000080880",
        "8888888888888888880800",
        "0000000000000000000808",
    ),
    4: (
        "80888888888888888888808888",
        "80800000000000000000800000",
        "80808888888888888880888880",
        "00808000000000000080000080",
        "88808088888888888088888080",
        "80008080000000008000008080",
        "80888080888888808888808080",
        "80800080800000800000808080",
        "80808880808880888880808080",
        "80808000808080000080808080",
        "80808088808088888080808080",
        "80808080008000008080808080",
        "80808088880888808080808080",
        "80808000008000808080808080",
        "80808888808088808080808080",
        "80800000808080008080808080",
        "80888880888080888080808080",
        "80000080000080800080808080",
        "88888088888880808880808080",
        "00008000000000808000808080",
        "88808888888888808088808080",
        "00800000000000008080008080",
        "80888888888888888080888080",
        "80000000000000000080800080",
        "88888888888888888880808880",
        "00000000000000000000808000",
    ),
}

_SOURCE_CENTERS = {
    1: (4, 5),
    2: (7, 6),
    3: (9, 9),
    4: (12, 10),
}

def _checked_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train", 2, 10)
    test_count = _checked_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif train_count is None:
        assert test_count is not None
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif test_count is None:
        largest_test_count = min(4, train_count - 1)
        if largest_test_count < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, largest_test_count + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _array_key(array: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(array.astype(np.int64))
    return normalized.shape, normalized.tobytes()


def _detect_seed(
    input_grid: np.ndarray,
) -> tuple[int, int, int, int, int]:
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2

    background_index = int(np.argmax(counts))
    assert counts[background_index] > counts[1 - background_index]

    background = int(values[background_index])
    foreground = int(values[1 - background_index])
    foreground_mask = input_grid == foreground

    foreground_count = int(np.count_nonzero(foreground_mask))
    assert foreground_count % 4 == 0
    arm_length = foreground_count // 4
    assert 1 <= arm_length <= 6

    row_counts = foreground_mask.sum(axis=1)
    col_counts = foreground_mask.sum(axis=0)
    center_rows = np.flatnonzero(row_counts == 2 * arm_length)
    center_cols = np.flatnonzero(col_counts == 2 * arm_length)
    assert len(center_rows) == 1
    assert len(center_cols) == 1

    center_row = int(center_rows[0])
    center_col = int(center_cols[0])
    assert input_grid[center_row, center_col] == background

    expected_mask = np.zeros_like(foreground_mask)
    for distance in range(1, arm_length + 1):
        assert 0 <= center_row - distance
        assert center_row + distance < input_grid.shape[0]
        assert 0 <= center_col - distance
        assert center_col + distance < input_grid.shape[1]

        expected_mask[center_row - distance, center_col] = True
        expected_mask[center_row + distance, center_col] = True
        expected_mask[center_row, center_col - distance] = True
        expected_mask[center_row, center_col + distance] = True

    assert np.array_equal(foreground_mask, expected_mask)
    return (
        background,
        foreground,
        arm_length,
        center_row,
        center_col,
    )


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0
    assert int(grid.max()) <= 9


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False
        for first_pair, second_pair in zip(
            first[split_name], second[split_name]
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False
    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    signature = []
    for split_name in ("train", "test"):
        signature.append(split_name)
        for pair in puzzle[split_name]:
            signature.append(_array_key(pair["input"]))
            signature.append(_array_key(pair["output"]))
    return tuple(signature)


def _seed_grid(length, shape, center, background, foreground):
    grid = np.full(shape, background, dtype=np.int8)
    y, x = center
    for d in range(1, length + 1):
        grid[y-d, x] = grid[y+d, x] = foreground
        grid[y, x-d] = grid[y, x+d] = foreground
    return grid


def _winding(length, shape, center, handedness):
    """Walk four stepped paths on the infinite plane, clipping segments only.

    The north path extends to R=max(2,L), then repeatedly travels forward
    2+4k, right R-2, and forward R. Its forward direction turns right after
    each triple. The other paths are its quarter turns. A reflected puzzle
    changes every right turn to a left turn.
    """
    height, width = shape
    cy, cx = center
    radius = max(2, length)
    mask = np.zeros(shape, dtype=bool)

    def paint(y0, x0, y1, x1):
        y0, y1 = y0 + cy, y1 + cy
        x0, x1 = x0 + cx, x1 + cx
        if y0 == y1 and 0 <= y0 < height:
            lo, hi = max(0, min(x0, x1)), min(width-1, max(x0, x1))
            if lo <= hi:
                mask[y0, lo:hi+1] = True
        elif x0 == x1 and 0 <= x0 < width:
            lo, hi = max(0, min(y0, y1)), min(height-1, max(y0, y1))
            if lo <= hi:
                mask[lo:hi+1, x0] = True

    distance = max(cy, cx, height-1-cy, width-1-cx)
    # A complete four-stage cycle moves the path outward eight cells.
    # After these cycles even its inward jog is outside the bounding square.
    stages = 4 * (distance // 8 + 2)
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        y, x = radius * dy, radius * dx
        paint(dy, dx, y, x)
        dy, dx = handedness * dx, -handedness * dy
        for k in range(stages):
            for vy, vx, steps in (
                (dy, dx, 4*k+2),
                (handedness*dx, -handedness*dy, radius-2),
                (dy, dx, radius),
            ):
                ny, nx = y + steps*vy, x + steps*vx
                paint(y, x, ny, nx)
                y, x = ny, nx
            dy, dx = handedness * dx, -handedness * dy
    return mask


def _analytic_winding(length, shape, center, handedness):
    """Separate cell-membership calculation with closed-form stage starts.

    No path walking, accumulated endpoints, renderer, or bitmap fixtures are
    used here. Each cell is tested against line intervals in four rotated
    coordinate systems. This guards the procedural renderer against drift.
    """
    r = max(2, length)
    yy, xx = np.indices(shape)
    yy = yy - center[0]
    xx = handedness * (xx - center[1])
    result = np.zeros(shape, dtype=bool)
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    for _ in range(4):
        result |= (xx == 0) & (yy >= -r) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-r-8*q, -8*q), (-2-8*q, r+8*q+2),
                      (r+8*q+4, 8*q+4), (8*q+6, -r-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                starts, ((0, 1), (1, 0), (0, -1), (-1, 0))
            )):
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                a = 16*q + 4*s + 2
                result |= ((across == 0) & (along >= 0) & (along <= a))
                result |= ((along == a) & (across >= 0) & (across <= r-2))
                result |= ((across == r-2) & (along >= a) & (along <= a+r))
        yy, xx = xx, -yy
    return result


def _render_pair(length, shape, center, background, foreground, handedness):
    a = _seed_grid(length, shape, center, background, foreground)
    b = np.where(_winding(length, shape, center, handedness),
                 foreground, background).astype(np.int8)
    return {"input": a, "output": b}


def _source_pairs():
    pairs = []
    for length, rows in _SOURCE_OUTPUT_ROWS.items():
        b = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        a = _seed_grid(length, b.shape, _SOURCE_CENTERS[length], 0, 8)
        pairs.append({"input": a, "output": b})
    return pairs


def _infer_handedness(train_pairs):
    candidates = {1, -1}
    for pair in train_pairs:
        a, b = pair["input"], pair["output"]
        bg, fg, length, y, x = _detect_seed(a)
        candidates &= {hand for hand in (1, -1) if np.array_equal(
            b, np.where(_analytic_winding(length, a.shape, (y, x), hand), fg, bg)
        )}
    assert len(candidates) == 1, "training must reveal one consistent handedness"
    return candidates.pop()


def _predict(train_pairs, test_input):
    """Predict from demonstrations and an input; no test labels or metadata."""
    hand = _infer_handedness(train_pairs)
    bg, fg, length, y, x = _detect_seed(test_input)
    return np.where(_analytic_winding(length, test_input.shape, (y, x), hand),
                    fg, bg).astype(np.int8)


def _length_plan(rng, ntrain, ntest, regime):
    if regime == "extrapolate":
        cap = 3 if ntrain == 2 else int(rng.integers(3, 6))
        train = [2, 3] if ntrain == 2 else (
            [1, 2, 3] if cap == 3 else [2, cap-1, cap])
        pool = list(range(1, cap+1))
        test = [cap+1] * ntest
    elif regime == "interpolate":
        cap = int(rng.integers(5, 7))
        missing = int(rng.integers(4, cap))
        train = [2, 3, cap]
        pool = [n for n in range(1, cap+1) if n != missing]
        test = [missing] + [int(rng.choice([missing, cap, 2])) for _ in range(ntest-1)]
    else:
        train = [int(x) for x in rng.choice(np.arange(1, 7), 2, replace=False)]
        pool = list(range(1, 7))
        test = []
    while len(train) < ntrain:
        train.append(int(rng.choice(pool)))
    rng.shuffle(train)
    # Put two distinct lengths first, for visible demonstrations and optional
    # matched canvas sizes. The remaining pair order is still randomized.
    other = next(i for i in range(1, len(train)) if train[i] != train[0])
    train[1], train[other] = train[other], train[1]
    if not test:
        choices = sorted(set(train))
        test = [int(rng.choice(choices)) for _ in range(ntest)]
        if ntest > 1 and rng.random() < .8:
            test[1] = int(rng.choice([n for n in choices if n != test[0]]))
    return train, test


def _sample_shape(rng, length, low, high, generous=False):
    minimum = max(low, 2*length+1, 7)
    if generous:
        minimum = max(minimum, 2*max(2, length)+5)
    assert minimum <= high
    kind = int(rng.integers(10))
    if kind == 0:
        return high, high
    if kind <= 2:
        side = int(rng.integers(minimum, high+1))
        return side, side
    if kind <= 5:
        short = int(rng.integers(minimum, min(high, max(minimum, 14))+1))
        long = int(rng.integers(max(short, high-6), high+1))
        return (short, long) if rng.random() < .5 else (long, short)
    return tuple(int(x) for x in rng.integers(minimum, high+1, size=2))


def _sample_center(rng, length, shape, generous=False):
    padding = 2 if generous else 0
    low = length + padding
    hi = (shape[0]-1-low, shape[1]-1-low)
    mode = int(rng.integers(10))
    if mode <= 1:
        return tuple((side-1)//2 for side in shape)
    if mode <= 4:
        # Four corners, with independent zero-to-two-cell tip margins.
        return tuple(int(rng.choice([low, upper])) if rng.random() < .6
                     else int(rng.choice([min(low+2, upper), max(low, upper-2)]))
                     for upper in hi)
    pos = [int(rng.integers(low, upper+1)) for upper in hi]
    if mode <= 6:
        axis = int(rng.integers(2))
        pos[axis] = int(rng.choice([low, hi[axis]]))
    return tuple(pos)


def _palettes(rng, ntrain, ntest):
    mode = int(rng.integers(7))
    first = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    if mode == 0:
        return [first] * (ntrain + ntest)
    pool = [int(c) for c in rng.permutation(10)]
    training_pool = pool[:6] if mode == 1 else pool
    pairs = [tuple(int(c) for c in rng.choice(training_pool, 2, replace=False))
             for _ in range(ntrain)]
    pairs[1] = pairs[0][::-1]  # Both roles change, and color identity is irrelevant.
    pairs.extend(tuple(int(c) for c in rng.choice(pool[6:] if mode == 1 else pool,
                                                2, replace=False))
                 for _ in range(ntest))
    return pairs


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    hand = int(rng.choice([1, -1], p=[.7, .3]))
    regimes = ["extrapolate", "familiar", "compact"]
    if ntrain >= 3:
        regimes.append("interpolate")
    regime = str(rng.choice(regimes))
    train_lengths, test_lengths = _length_plan(rng, ntrain, ntest, regime)
    palettes = _palettes(rng, ntrain, ntest)
    train_high = int(rng.integers(21, 28)) if regime == "extrapolate" else 30
    train_low = int(rng.integers(20, 24)) if regime == "compact" else 7
    matched_shape = None
    if rng.random() < .4:
        matched_shape = _sample_shape(rng, max(train_lengths[:2]), train_low,
                                      train_high, generous=True)
    puzzle = {"train": [], "test": []}
    used = set()
    source_keys = {(_array_key(p["input"]), _array_key(p["output"]))
                   for p in _source_pairs()}
    for index, length in enumerate(train_lengths + test_lengths):
        is_train = index < ntrain
        split = "train" if is_train else "test"
        # Each demonstrated scale exposes the first complete jog; tests may
        # clip arbitrarily because the continuation has already been taught.
        generous = is_train
        if is_train:
            low, high = train_low, train_high
        elif regime == "extrapolate":
            low = 1 + max(max(p["input"].shape) for p in puzzle["train"])
            high = 30
        elif regime == "compact":
            low = 7
            high = min(min(p["input"].shape) for p in puzzle["train"])
        else:
            low, high = 7, 30
        bg, fg = palettes[index]
        shape = matched_shape if is_train and index < 2 and matched_shape else (
            _sample_shape(rng, length, low, high, generous))
        if index == ntrain+ntest-1 and all(
            p["input"].shape == shape for s in ("train", "test") for p in puzzle[s]
        ):
            minimum = max(low, 2*length+1, 7)
            side = shape[0]-1 if shape[0] > minimum else shape[0]+1
            assert minimum <= side <= high
            shape = (side, shape[1])
        # Most training pairs show a complete local turn. Later repeated
        # scales may use the full placement range, including touching edges.
        seen_lengths = set(train_lengths[:index]) if is_train else set()
        safe_center = is_train and length not in seen_lengths
        center = _sample_center(rng, length, shape, safe_center)
        # Try the sampled location first, then a deterministic finite walk
        # through every feasible center. There are far more than 14 choices.
        padding = 2 if safe_center else 0
        lo = length + padding
        centers = [center] + [(y, x) for y in range(lo, shape[0]-lo)
                              for x in range(lo, shape[1]-lo) if (y, x) != center]
        for center in centers:
            pair = _render_pair(length, shape, center, bg, fg, hand)
            ikey, okey = _array_key(pair["input"]), _array_key(pair["output"])
            if ikey not in used and (ikey, okey) not in source_keys:
                used.add(ikey)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("feasible placement space exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    source_keys = {(_array_key(p["input"]), _array_key(p["output"]))
                   for p in _source_pairs()}
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            _validate_grid(a)
            _validate_grid(b)
            assert a.shape == b.shape
            ikey, okey = _array_key(a), _array_key(b)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            assert (ikey, okey) not in source_keys
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            bg, fg, length, y, x = _detect_seed(a)
            assert set(int(v) for v in np.unique(b)) == {bg, fg}
            assert np.all(b[a == fg] == fg) and b[y, x] == bg
            assert np.any((a == bg) & (b == fg))
            info[split].append((length, a.shape, (y, x)))
    # Infer conventions exclusively from training; only then inspect labels.
    hand = _infer_handedness(puzzle["train"])
    for pair in puzzle["test"]:
        a = pair["input"]
        bg, fg, length, y, x = _detect_seed(a)
        prediction = np.where(_analytic_winding(length, a.shape, (y, x), hand), fg, bg)
        assert np.array_equal(prediction, pair["output"]), "incorrect continuation"
    train_lengths = {item[0] for item in info["train"]}
    assert len(train_lengths) >= 2
    # A visible first jog at every demonstrated scale, including the special
    # radius-two completion whenever a length-one test is present.
    for length in train_lengths:
        assert any(l == length and min(y-l, x-l, h-1-y-l, w-1-x-l) >= 2
                   for l, (h, w), (y, x) in info["train"])
    for length, (height, width), _ in info["test"]:
        if length == 1:
            assert 1 in train_lengths
        if length not in train_lengths:
            assert length <= max(train_lengths)+1
            assert len(train_lengths & set(range(2, 7))) >= 2
        if length > max(train_lengths):
            # The source-style regime combines the next spoke length with
            # larger dimensions. Other regimes use seen or interior lengths.
            assert height > max(shape[0] for _, shape, _ in info["train"])
            assert width > max(shape[1] for _, shape, _ in info["train"])
    assert len({shape for _, shape, _ in info["train"] + info["test"]}) >= 2


def _measure_puzzles(puzzles):
    """Review measurements reconstructed from returned grids, not RNG plans."""
    from collections import Counter

    totals = Counter()
    histograms = {name: Counter() for name in (
        "train_counts", "test_counts", "palettes_per_puzzle", "shapes_per_puzzle",
        "train_lengths", "test_lengths", "handedness", "length_shape_counts")}
    shapes, layouts, palettes = set(), set(), set()
    length_shapes = {n: set() for n in range(1, 7)}
    margins = {s: {"touch": [0]*4, "near": [0]*4, "corners": [0]*4,
                   "centered": 0, "unequal": 0, "generous": 0,
                   "tall": 0, "wide": 0} for s in ("train", "test")}
    per_puzzle = []
    for puzzle in puzzles:
        data = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, fg, length, y, x = _detect_seed(pair["input"])
                h, w = pair["input"].shape
                m = (y-length, h-1-y-length, x-length, w-1-x-length)
                data[split].append((length, h, w, bg, fg))
                shapes.add((h, w))
                layouts.add((length, h, w, y, x))
                palettes.add((bg, fg))
                length_shapes[length].add((h, w))
                histograms[split+"_lengths"][length] += 1
                totals["pairs"] += 1
                totals["full_30_square"] += int((h, w) == (30, 30))
                totals["small_cross_large_canvas"] += int(length <= 2 and h*w >= 600)
                totals["large_cross_tight_canvas"] += int(length >= 3 and min(h, w) <= 2*length+5)
                totals["combined_small_elongated_near_edge"] += int(
                    length <= 2 and max(h/w, w/h) >= 2 and min(m) <= 2)
                stats = margins[split]
                for side in range(4):
                    stats["touch"][side] += int(m[side] == 0)
                    stats["near"][side] += int(m[side] <= 2)
                for corner, (a, b) in enumerate(((0, 2), (0, 3), (1, 2), (1, 3))):
                    stats["corners"][corner] += int(m[a] <= 2 and m[b] <= 2)
                stats["centered"] += int(abs(2*y-h+1) <= 1 and abs(2*x-w+1) <= 1)
                stats["unequal"] += int(max(m)-min(m) >= 10)
                stats["generous"] += int(min(m) >= 5)
                stats["tall"] += int(h >= 2*w)
                stats["wide"] += int(w >= 2*h)
        train, test = data["train"], data["test"]
        tl, vl = {p[0] for p in train}, {p[0] for p in test}
        ps = {(p[3], p[4]) for p in train+test}
        sh = {(p[1], p[2]) for p in train+test}
        train_colors = {c for p in train for c in p[3:]}
        swapped = any((fg, bg) in ps for bg, fg in ps)
        novel = any(c not in train_colors for p in test for c in p[3:])
        same_size_lengths = any(len({p[0] for p in train+test if p[1:3] == shape}) > 1
                                for shape in sh)
        extrapolate = min(vl) > max(tl) and all(
            p[1] > max(t[1] for t in train) and p[2] > max(t[2] for t in train) for p in test)
        interpolate = any(min(tl) < n < max(tl) and n not in tl for n in vl)
        smaller = max(p[1]*p[2] for p in test) < min(p[1]*p[2] for p in train)
        hand = _infer_handedness(puzzle["train"])
        histograms["handedness"][hand] += 1
        histograms["train_counts"][len(train)] += 1
        histograms["test_counts"][len(test)] += 1
        histograms["palettes_per_puzzle"][len(ps)] += 1
        histograms["shapes_per_puzzle"][len(sh)] += 1
        flags = {"common_palette": len(ps) == 1, "varied_palettes": len(ps) > 1,
                 "role_swaps": swapped, "novel_test_colors": novel,
                 "same_size_different_lengths": same_size_lengths,
                 "source_extrapolation": extrapolate, "interpolation": interpolate,
                 "demonstrated_test_lengths": vl <= tl,
                 "all_tests_smaller": smaller, "mixed_test_lengths": len(vl) > 1,
                 "tests_within_training_dimensions": all(
                     p[1] <= max(t[1] for t in train) and p[2] <= max(t[2] for t in train)
                     for p in test)}
        totals.update({key: int(value) for key, value in flags.items()})
        per_puzzle.append({"train_lengths": sorted(tl), "test_lengths": sorted(vl),
                           "palettes": len(ps), "shapes": len(sh), "handedness": hand,
                           **flags})
    totals["puzzles"] = len(puzzles)
    totals["distinct_shapes"] = len(shapes)
    totals["distinct_layouts"] = len(layouts)
    totals["distinct_ordered_palettes"] = len(palettes)
    for length, observed in length_shapes.items():
        histograms["length_shape_counts"][length] = len(observed)
    return {"totals": dict(totals), "histograms": {k: dict(v) for k, v in histograms.items()},
            "margins": margins, "per_puzzle": per_puzzle}


def self_test() -> None:
    # Fixtures are checked independently of generation, including the source
    # held-out prediction. They are never used to synthesize a new winding.
    source = _source_pairs()
    for pair in source:
        _, fg, length, y, x = _detect_seed(pair["input"])
        for calculation in (_winding, _analytic_winding):
            assert np.array_equal(calculation(length, pair["input"].shape, (y, x), 1),
                                  pair["output"] == fg)
    assert source[2]["output"][21, 21] == 8
    assert np.array_equal(_predict(source[:3], source[3]["input"].copy()), source[3]["output"])
    assert np.array_equal(_winding(1, (30, 30), (15, 15), 1),
                          _winding(2, (30, 30), (15, 15), 1))
    for length in range(1, 7):
        for hand in (1, -1):
            for shape, center in (((30, 30), (length, length)),
                                   ((25, 29), (24-length, 28-length)),
                                   ((2*length+5, 30), (length+2, 15))):
                mask = _winding(length, shape, center, hand)
                assert np.array_equal(mask, _analytic_winding(length, shape, center, hand))
                rotated_center = (shape[1]-1-center[1], center[0])
                assert np.array_equal(np.rot90(mask), _winding(
                    length, shape[::-1], rotated_center, hand))
                assert np.array_equal(np.fliplr(mask), _winding(
                    length, shape, (center[0], shape[1]-1-center[1]), -hand))
                assert not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
                # Cropping an oversized continuation must keep re-entry cells.
                big = _analytic_winding(length, (70, 70), (35, 35), hand)
                top, left = 35-center[0], 35-center[1]
                assert np.array_equal(mask, big[top:top+shape[0], left:left+shape[1]])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    measurements = _measure_puzzles(puzzles)
    t, h, margins = measurements["totals"], measurements["histograms"], measurements["margins"]
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert len(h["train_counts"]) >= 4 and len(h["test_counts"]) >= 3
    assert t["distinct_shapes"] >= 90 and t["distinct_layouts"] >= 200
    assert t["distinct_ordered_palettes"] >= 40
    assert min(h["length_shape_counts"].values()) >= 10
    assert set(h["train_lengths"]) == set(h["test_lengths"]) == set(range(1, 7))
    assert set(h["handedness"]) == {-1, 1}
    for feature in ("common_palette", "varied_palettes", "novel_test_colors", "role_swaps",
                    "source_extrapolation", "interpolation", "demonstrated_test_lengths",
                    "all_tests_smaller", "mixed_test_lengths", "same_size_different_lengths",
                    "tests_within_training_dimensions", "full_30_square",
                    "small_cross_large_canvas", "large_cross_tight_canvas",
                    "combined_small_elongated_near_edge"):
        assert t[feature] > 0, feature
    for split in ("train", "test"):
        assert min(margins[split]["touch"]) > 0
        assert min(margins[split]["corners"]) > 0
        for feature in ("centered", "unequal", "generous", "tall", "wide"):
            assert margins[split][feature] > 0, (split, feature)
    for seed, train_count, test_count in ((123456, None, None), (654321, 7, 3),
                                         (2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
                                         (3000, None, 1), (3001, None, 4),
                                         (3002, 2, None), (3003, 10, None)):
        a = generate_puzzle(seed, train_count, test_count)
        b = generate_puzzle(seed, train_count, test_count)
        assert _puzzles_equal(a, b)
        if train_count is not None:
            assert len(a["train"]) == train_count
        if test_count is not None:
            assert len(a["test"]) == test_count
        validate_generated_puzzle(a)
    for train_count, test_count in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2),
                                    (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(4000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for puzzle in puzzles:
        # Only copies of demonstrations and one test input enter prediction.
        training = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
        for pair in puzzle["test"]:
            prediction = _predict(training, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    for mutation in ("pixel", "handedness"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        bg, fg, length, y, x = _detect_seed(pair["input"])
        if mutation == "pixel":
            row, col = next((r, c) for r, c in np.argwhere(pair["input"] == bg)
                            if (r, c) != (y, x))
            pair["output"][row, col] = fg if pair["output"][row, col] == bg else bg
        else:
            hand = _infer_handedness(bad["train"])
            pair["output"] = np.where(_winding(length, pair["input"].shape, (y, x), -hand), fg, bg)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect test output accepted")
