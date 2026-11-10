"""Connected, edge-clipped ribbon completion puzzles."""

import hashlib
import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    "e7801e7a0e62b41b8ca6f9cc3dd1c6b69949c3f8fb590ed8bbde31c1b7327b67",
    "30c035b465f977481473f92b1acc7074f442257b9d53694d691a385436b6fdd4",
    "fb3e7259f68ddd29fb4fd6841779dd1ae59a2769b9d6233c07348d215c9499c7",
    "f7a3a360a66a377b8f5126543936ffc51a696eb5dc08f0b5b1a3f34cf4ee3959",
})
_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        code = abs(int(seed))
        num_test = 1 + code % 4
        lower = max(2, num_test + 1)
        num_train = lower + (code // 4) % (11 - lower)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _boundary(mask):
    """Visible internal sides; the canvas itself closes clipped ends and sides."""
    edge = np.zeros_like(mask)
    edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    return edge


def _scan_band(rng, h, w, zone, clipped, vigorous):
    lo, hi = zone
    available = hi - lo + 1
    min_gap = 4 if available < 12 else 5
    max_gap = min(10 if vigorous else 7, available - 2)
    gap = int(rng.integers(min_gap, max_gap + 1))
    left = int(rng.integers(lo + 1, hi - gap + 1))
    knots = [0]
    while knots[-1] < h - 1:
        knots.append(min(h - 1, knots[-1] + int(rng.integers(3, 8))))
    left_targets = [left]
    gap_targets = [gap]
    direction = int(rng.choice((-1, 1)))
    for index in range(1, len(knots)):
        if vigorous:
            if index % 2 == 0:
                direction *= -1
            delta = direction * int(rng.integers(3, 9))
        else:
            delta = int(rng.integers(-3, 4))
        gap = int(np.clip(gap + int(rng.integers(-2, 3)), min_gap, max_gap))
        left = int(np.clip(left + delta, lo + 1, hi - gap))
        left_targets.append(left)
        gap_targets.append(gap)
    xs = np.arange(h)
    lefts = np.rint(np.interp(xs, knots, left_targets)).astype(int)
    gaps = np.rint(np.interp(xs, knots, gap_targets)).astype(int)
    rights = lefts + gaps
    if clipped and zone == (0, w - 1):
        side = int(rng.integers(0, 2))
        at_start = bool(rng.integers(0, 2))
        count = max(3, int(rng.integers(3, max(4, h // 3 + 1))))
        indices = range(count) if at_start else range(h - count, h)
        for i in indices:
            if side == 0:
                lefts[i] = 0
            else:
                rights[i] = w - 1
        # A gradual return keeps the visible bank followable.
        if at_start and count + 2 < h:
            for j in range(2):
                i = count + j
                if side == 0:
                    lefts[i] = min(lefts[i], j + 1)
                else:
                    rights[i] = max(rights[i], w - 2 - j)
        elif not at_start and count + 2 < h:
            for j in range(2):
                i = h - count - 1 - j
                if side == 0:
                    lefts[i] = min(lefts[i], j + 1)
                else:
                    rights[i] = max(rights[i], w - 2 - j)
    mask = np.zeros((h, w), dtype=bool)
    for row in range(h):
        mask[row, lefts[row]:rights[row] + 1] = True
    return mask


def _rect(mask, r0, r1, c0, c1):
    mask[max(0, r0):min(mask.shape[0], r1), max(0, c0):min(mask.shape[1], c1)] = True


def _turn_band(rng, h, w, dogleg):
    mask = np.zeros((h, w), dtype=bool)
    radius = int(rng.integers(2, min(5, h // 6 + 1, w // 6 + 1)))
    if dogleg:
        turn = int(rng.integers(h // 3, 2 * h // 3 + 1))
        c0 = int(rng.integers(radius + 2, max(radius + 3, w // 3 + 1)))
        c1 = int(rng.integers(max(c0 + 2 * radius + 4, 2 * w // 3), w - radius - 1))
        _rect(mask, 0, turn + radius + 1, c0 - radius, c0 + radius + 1)
        _rect(mask, turn - radius, turn + radius + 1, c0, c1 + 1)
        _rect(mask, turn - radius, h, c1 - radius, c1 + radius + 1)
    else:
        turn = int(rng.integers(h // 3, 2 * h // 3 + 1))
        col = int(rng.integers(w // 3, 2 * w // 3 + 1))
        _rect(mask, 0, turn + radius + 1, col - radius, col + radius + 1)
        _rect(mask, turn - radius, turn + radius + 1, col, w)
    return np.rot90(mask, int(rng.integers(0, 4))).copy()


def _make_mask(rng, h, w, mode, clipped, orientation):
    if mode in ("turn", "dogleg"):
        return _turn_band(rng, h, w, mode == "dogleg")
    if orientation == "horizontal":
        h, w = w, h
    if mode == "multi":
        if w < 24:
            raise ValueError("multi ribbon needs transverse space")
        left_hi = int(rng.integers(8, 11))
        right_lo = int(rng.integers(w - 10, w - 7))
        if right_lo - left_hi < 7:
            raise ValueError("ribbons too close")
        mask = _scan_band(rng, h, w, (0, left_hi), False, False)
        mask |= _scan_band(rng, h, w, (right_lo, w - 1), False, False)
    else:
        mask = _scan_band(rng, h, w, (0, w - 1), clipped, mode == "winding")
    return mask.T.copy() if orientation == "horizontal" else mask


def _components(binary, neighbors):
    h, w = binary.shape
    seen = np.zeros_like(binary)
    result = []
    for raw_r, raw_c in np.argwhere(binary):
        r, c = int(raw_r), int(raw_c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(cells)
    return result


def _predict_from_input(input_grid):
    """Input-only reconstruction by the two long banks of each narrow region."""
    values, counts = np.unique(input_grid, return_counts=True)
    if len(values) != 3:
        return None
    bg = int(values[np.argmax(counts)])
    other = [int(v) for v in values if v != bg]
    records = []
    for color in other:
        components = _components(input_grid == color, _NEIGHBORS_8)
        long = [part for part in components if len(part) >= max(8, min(input_grid.shape) // 3)]
        records.append((sum(len(part) for part in long), color, long))
    records.sort(reverse=True, key=lambda item: item[0])
    if records[0][0] == 0 or records[1][0] != 0:
        return None
    _, paint, banks = records[0]
    if len(banks) < 2:
        return None
    bank_id = np.full(input_grid.shape, -1, dtype=np.int16)
    for index, part in enumerate(banks):
        for cell in part:
            bank_id[cell] = index
    output = input_grid.copy()
    selected = 0
    # Protected marks are overlays.  Traverse through them when identifying
    # an enclosed ribbon, then recolor background cells only.
    for region in _components(input_grid != paint, _NEIGHBORS_4):
        contact = {}
        for r, c in region:
            for dr, dc in _NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1]:
                    label = int(bank_id[nr, nc])
                    if label >= 0:
                        contact[label] = contact.get(label, 0) + 1
        strong = sorted((v for v in contact.values() if v >= 3), reverse=True)
        if len(strong) < 2:
            continue
        # This separates paired banks from the broad space between ribbons.
        if len(region) / strong[1] > 8.5:
            continue
        for cell in region:
            if input_grid[cell] == bg:
                output[cell] = paint
        selected += 1
    return (output, bg, paint, selected) if selected else None


def _random_mark(rng, max_length):
    kind = int(rng.integers(0, 4))
    length = int(rng.integers(1, max_length + 1))
    if kind == 0:
        return [(0, k) for k in range(length)]
    if kind == 1:
        return [(k, 0) for k in range(length)]
    if kind == 2:
        return [(k, k) for k in range(length)]
    return [(0, k) for k in range(max(1, length - 1))] + [(1, max(0, length - 2))]


def _place_marks(rng, grid, mask, boundary, background, paint, protected):
    h, w = grid.shape
    # One protected mark inside the fill makes preservation visible in every pair.
    groups = int(rng.integers(1, 6))
    for group in range(groups):
        inside = group == 0 or bool(rng.integers(0, 2))
        length_limit = 5 if group == 0 or bool(rng.integers(0, 3)) else 2
        for _ in range(100):
            shape = _random_mark(rng, length_limit)
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if all(0 <= y < h and 0 <= x < w and bool(mask[y, x]) == inside and grid[y, x] == background for y, x in cells):
                for cell in cells:
                    grid[cell] = protected
                break
    fragments = int(rng.integers(0, 8))
    for _ in range(fragments):
        for attempt in range(70):
            shape = _random_mark(rng, 4)
            if attempt < 25:
                ys, xs = np.where(boundary)
                k = int(rng.integers(0, len(ys)))
                r = int(ys[k]) + int(rng.integers(-3, 4))
                c = int(xs[k]) + int(rng.integers(-3, 4))
            else:
                r = int(rng.integers(0, h))
                c = int(rng.integers(0, w))
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if not all(0 <= y < h and 0 <= x < w and not mask[y, x] and grid[y, x] == background for y, x in cells):
                continue
            around = {(y + dy, x + dx) for y, x in cells for dy, dx in _NEIGHBORS_8}
            if any(0 <= y < h and 0 <= x < w and grid[y, x] == paint for y, x in around):
                continue
            for cell in cells:
                grid[cell] = paint
            break
    # Small protected clutter outside can be a bar or an irregular cluster.
    for _ in range(int(rng.integers(0, 5))):
        for _ in range(50):
            shape = _random_mark(rng, 5)
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if all(0 <= y < h and 0 <= x < w and not mask[y, x] and grid[y, x] == background for y, x in cells):
                for cell in cells:
                    grid[cell] = protected
                break


def _pair(rng, palette, mode, orientation, clipped, dimensions):
    h, w = dimensions
    mask = _make_mask(rng, h, w, mode, clipped, orientation)
    boundary = _boundary(mask)
    background, paint, protected = palette
    input_grid = np.full(mask.shape, background, dtype=np.int8)
    input_grid[boundary] = paint
    _place_marks(rng, input_grid, mask, boundary, background, paint, protected)
    output_grid = input_grid.copy()
    output_grid[mask & (input_grid == background)] = paint
    prediction = _predict_from_input(input_grid)
    if prediction is None or not np.array_equal(prediction[0], output_grid):
        return None
    if not np.any(input_grid != output_grid):
        return None
    return {"input": input_grid, "output": output_grid}


def _signature(array):
    return (array.shape, array.tobytes())


def _grid_check(array):
    assert isinstance(array, np.ndarray) and array.ndim == 2
    assert np.issubdtype(array.dtype, np.integer)
    assert all(1 <= n <= 30 for n in array.shape)
    assert np.all((0 <= array) & (array <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    roles = []
    training_has_hole = False
    training_has_turn = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            _grid_check(source)
            _grid_check(target)
            assert source.shape == target.shape
            key = _signature(source)
            assert key not in seen
            seen.add(key)
            if source.shape == (30, 30):
                assert hashlib.sha256(source.astype(np.int8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
            prediction = _predict_from_input(source)
            assert prediction is not None
            solved, bg, paint, count = prediction
            assert np.array_equal(solved, target) and count >= 1
            changed = source != target
            assert np.any(changed)
            assert np.all(source[changed] == bg)
            assert np.all(target[changed] == paint)
            assert np.array_equal(source[~changed & (source != bg)], target[~changed & (source != bg)])
            protected = next(int(v) for v in np.unique(source) if v not in (bg, paint))
            roles.append((bg, paint, protected))
            inside_holes = (source == protected) & (target == protected)
            assert np.any(inside_holes)
            if split == "train":
                training_has_hole |= bool(np.any(inside_holes & _adjacent_to_change(changed)))
                training_has_turn |= _has_turning_trace(source, paint)
    assert len(set(roles[:len(puzzle["train"])])) >= 2
    assert training_has_hole and training_has_turn
    # The rule and color-role evidence are in training; no fixed literal palette
    # or orientation is needed for test reconstruction.


def _adjacent_to_change(changed):
    near = changed.copy()
    near[1:, :] |= changed[:-1, :]
    near[:-1, :] |= changed[1:, :]
    near[:, 1:] |= changed[:, :-1]
    near[:, :-1] |= changed[:, 1:]
    return near


def _has_turning_trace(source, paint):
    for part in _components(source == paint, _NEIGHBORS_8):
        if len(part) < 12:
            continue
        rows = {r for r, _ in part}
        cols = {c for _, c in part}
        if len(part) > max(len(rows), len(cols)) + 4:
            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    base_palette = [int(x) for x in rng.choice(10, 3, replace=False)]
    reassigned_palette = (
        (base_palette[1], base_palette[2], base_palette[0])
        if bool(rng.integers(0, 2))
        else (base_palette[2], base_palette[0], base_palette[1])
    )
    turn_index = int(rng.integers(0, ntrain))
    turn_mode = str(rng.choice(("turn", "dogleg")))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train" and index == 0:
                palette = tuple(base_palette)
            elif split == "train" and index == 1:
                palette = reassigned_palette
            else:
                palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
            desired_mode = turn_mode if split == "train" and index == turn_index else None
            for attempt in range(150):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if desired_mode is None:
                    modes = ("scan", "winding", "turn", "dogleg", "multi")
                    probs = (0.18, 0.23, 0.19, 0.17, 0.23)
                    mode = str(child_rng.choice(modes, p=probs))
                else:
                    mode = desired_mode
                if mode == "multi":
                    h = int(child_rng.integers(24, 31))
                    w = int(child_rng.integers(26, 31))
                elif mode == "dogleg":
                    h = int(child_rng.integers(23, 31))
                    w = int(child_rng.integers(25, 31))
                else:
                    h = int(child_rng.integers(15, 31))
                    w = int(child_rng.integers(15, 31))
                if child_rng.random() < 0.13:
                    h = w = 30
                orientation = str(child_rng.choice(("vertical", "horizontal")))
                clipped = bool(child_rng.random() < 0.42)
                try:
                    pair = _pair(child_rng, palette, mode, orientation, clipped, (h, w))
                except (ValueError, RuntimeError):
                    continue
                if pair is None:
                    continue
                key = _signature(pair["input"])
                if key in used:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("unable to construct a uniquely reconstructible ribbon")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, palettes, role_tuples, layouts = set(), set(), set(), set()
    single_train = multi_train = multi_test = horizontal_test = 0
    full_size = clipped_edge = clustered_protected = 0
    near_clean = clutter_rich = turning_traces = changing_spans = 0
    within_puzzle_count_variety = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_counts = set()
        training_roles = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                sizes.add(source.shape)
                prediction = _predict_from_input(source)
                assert prediction is not None and np.array_equal(prediction[0], target)
                _, background, paint, region_count = prediction
                protected = next(int(v) for v in np.unique(source) if v not in (background, paint))
                role_tuples.add((background, paint, protected))
                if split == "train":
                    training_roles.append((background, paint, protected))
                pair_counts.add(region_count)
                palettes.add(tuple(int(x) for x in np.unique(source)))
                layouts.add((source.shape, (source != target).tobytes()))
                full_size += source.shape == (30, 30)
                changed = source != target
                edge_contacts = sum(bool(x) for x in (
                    np.any(changed[0]), np.any(changed[-1]),
                    np.any(changed[:, 0]), np.any(changed[:, -1]),
                ))
                clipped_edge += edge_contacts >= 3
                paint_parts = _components(source == paint, _NEIGHBORS_8)
                short_fragments = sum(len(part) < 8 for part in paint_parts)
                near_clean += short_fragments <= 1
                clutter_rich += short_fragments >= 5
                protected_parts = _components(source == protected, _NEIGHBORS_8)
                clustered_protected += max(map(len, protected_parts)) >= 3
                turning_traces += _has_turning_trace(source, paint)
                spans = [int(np.sum(changed[row])) for row in range(source.shape[0]) if np.any(changed[row])]
                changing_spans += max(spans) - min(spans) >= 3
                if split == "train":
                    single_train += region_count == 1
                    multi_train += region_count >= 2
                else:
                    multi_test += region_count >= 2
                    horizontal_test += bool(np.any(changed[:, 0]) and np.any(changed[:, -1]))
        assert len(set(training_roles)) >= 2
        assert any(a[1] in (b[0], b[2]) for a in training_roles for b in training_roles if a != b)
        within_puzzle_count_variety += len(pair_counts) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 35 and len(palettes) >= 20 and len(role_tuples) >= 80
    assert len(layouts) >= 100 and full_size >= 5
    assert clipped_edge >= 20 and clustered_protected >= 100
    assert near_clean >= 15 and clutter_rich >= 30
    assert turning_traces >= 100 and changing_spans >= 75
    assert single_train >= 100 and multi_train >= 3 and multi_test >= 2
    assert horizontal_test >= 2 and within_puzzle_count_variety >= 10
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + i, a, b)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for i, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(21000 + i, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    first = generate_puzzle(99173, 6, 2)
    second = generate_puzzle(99173, 6, 2)
    assert all(np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"])
               for split in ("train", "test") for p, q in zip(first[split], second[split]))
    assert not np.array_equal(first["train"][0]["input"], generate_puzzle(99174, 6, 2)["train"][0]["input"])
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
