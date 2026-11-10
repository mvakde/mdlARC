import numpy as np


_SOURCE_INPUT_SIGNATURES = frozenset({
    (12, 13, 0, 1, ((1, 8, 1, 9), (9, 12, 8, 11))),
    (12, 11, 0, 1, ((1, 4, 7, 10), (1, 5, 1, 5), (6, 12, 2, 7))),
    (15, 14, 0, 1, ((1, 9, 2, 10), (10, 15, 4, 11))),
    (15, 17, 0, 1, ((0, 6, 2, 13), (7, 15, 9, 16), (9, 12, 1, 6))),
})


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _separated(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _signature(grid, background, foreground, rectangles):
    return (grid.shape[0], grid.shape[1], background, foreground, tuple(sorted(rectangles)))


def _depth(rectangle):
    r0, r1, c0, c1 = rectangle
    return (min(r1 - r0, c1 - c0) - 1) // 2


def _shape(rng, height, width, count, depth_cap, kind):
    if kind == "deep":
        short = 2 * depth_cap + 1
        long = short + int(rng.integers(0, min(5, max(height, width) - short)))
        h, w = (short, long) if rng.integers(2) else (long, short)
        if h <= height - 2 and w <= width - 2:
            return h, w
        return short, short

    side_limit = min(2 * depth_cap + 2, 8 if count >= 4 else 11)
    long_limit = min(10 if count >= 4 else 15, max(height, width) - 2)
    if kind == "bar":
        short = int(rng.integers(1, 3))
        long = int(rng.integers(3, long_limit + 1))
        h, w = (short, long) if rng.integers(2) else (long, short)
    else:
        motif = int(rng.integers(0, 4))
        if motif == 0:
            side = int(rng.integers(3, min(side_limit, 9) + 1))
            h = side
            w = side + int(rng.integers(-1, 2))
        elif motif == 1:
            h = int(rng.integers(3, min(side_limit, 8) + 1))
            w = int(rng.integers(h + 1, max(h + 2, long_limit + 1)))
        elif motif == 2:
            w = int(rng.integers(3, min(side_limit, 8) + 1))
            h = int(rng.integers(w + 1, max(w + 2, long_limit + 1)))
        else:
            h = int(rng.integers(3, side_limit + 1))
            w = int(rng.integers(3, side_limit + 1))
    h = max(1, min(h, height - 2))
    w = max(1, min(w, width - 2))
    return h, w


def _place(rng, height, width, shapes, contact_mode):
    count = len(shapes)
    contacts = ["free"] * count
    if contact_mode == "internal":
        contacts = ["internal"] * count
    elif contact_mode == "mixed":
        edge_index = int(rng.integers(count))
        inner_index = (edge_index + int(rng.integers(1, count))) % count
        contacts[edge_index] = "corner" if rng.random() < 0.42 else "edge"
        contacts[inner_index] = "internal"
    elif contact_mode in ("edge", "corner"):
        contacts[int(rng.integers(count))] = contact_mode

    order = sorted(range(count), key=lambda i: (
        contacts[i] not in ("corner", "edge"),
        -(shapes[i][0] * shapes[i][1]),
    ))
    placed = []
    for i in order:
        h, w = shapes[i]
        candidates = []
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                r1, c1 = r0 + h, c0 + w
                touch_rows = r0 == 0 or r1 == height
                touch_cols = c0 == 0 or c1 == width
                contact = contacts[i]
                if contact == "internal" and (touch_rows or touch_cols):
                    continue
                if contact == "edge" and not (touch_rows or touch_cols):
                    continue
                if contact == "corner" and not (touch_rows and touch_cols):
                    continue
                rect = (r0, r1, c0, c1)
                if all(_separated(rect, other) for other in placed):
                    candidates.append(rect)
        if not candidates:
            return None
        placed.append(candidates[int(rng.integers(len(candidates)))])
    return sorted(placed)


def _make_pair(rng, palette, count, depth_cap, contact_mode,
               required_depth=0, force_bar=False):
    background, foreground, accent1, accent2 = palette
    if count == 1 and contact_mode == "mixed":
        contact_mode = "edge"
    for _ in range(180):
        lower = 9 if count == 1 else 12 if count == 2 else 15 if count == 3 else 19
        lower = max(lower, 2 * required_depth + 4)
        height = int(rng.integers(lower, 31))
        width = int(rng.integers(lower, 31))

        shapes = []
        comparable = count >= 3 and required_depth == 0 and rng.random() < 0.4
        if comparable:
            upper = 6 if count >= 4 else 8
            base_h = int(rng.integers(4, upper))
            base_w = int(rng.integers(4, upper))
            shapes = [(base_h + int(rng.integers(2)),
                       base_w + int(rng.integers(2))) for _ in range(count)]
        else:
            for i in range(count):
                if i == 0 and required_depth:
                    kind = "deep"
                elif i == 1 and force_bar:
                    kind = "bar"
                elif i == 0:
                    kind = "changing"
                else:
                    kind = "bar" if rng.random() < 0.22 else "changing"
                shapes.append(_shape(rng, height, width, count, depth_cap, kind))
        if not any(min(h, w) >= 3 for h, w in shapes):
            continue
        rectangles = _place(rng, height, width, shapes, contact_mode)
        if rectangles is None:
            continue
        if sum((r1-r0)*(c1-c0) for r0,r1,c0,c1 in rectangles) > 0.65 * height * width:
            continue

        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for r0, r1, c0, c1 in rectangles:
            input_grid[r0:r1, c0:c1] = foreground
            output_grid[r0:r1, c0:c1] = foreground
            for d in range(1, _depth((r0, r1, c0, c1)) + 1):
                color = accent1 if d % 2 else accent2
                output_grid[r0+d:r1-d, c0+d:c1-d] = color
        if _signature(input_grid, background, foreground, rectangles) in _SOURCE_INPUT_SIGNATURES:
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("failed to place separated rectangles")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    palette = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
    depth_cap = int(rng.choice((3, 4, 5), p=(0.65, 0.26, 0.09)))
    regime = int(rng.integers(0, 3))
    witness = int(rng.integers(num_train))
    internal_pair = int(rng.integers(num_train))
    mixed_pair = (internal_pair + int(rng.integers(1, num_train))) % num_train
    puzzle = {"train": [], "test": []}
    seen = set()

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                if index == witness:
                    objects = int(rng.integers(2, 5))
                elif index == mixed_pair:
                    objects = int(rng.integers(2, 6))
                else:
                    objects = int(rng.integers(1, 6))
                contact = ("internal" if index == internal_pair else
                           "mixed" if index == mixed_pair else
                           ("internal", "edge", "corner", "mixed")[int(rng.integers(4))])
                required = depth_cap if index == witness else 0
                bar = index == witness
            else:
                objects = int(rng.integers(1 if regime == 0 else 2,
                                           4 if regime == 0 else 6))
                contact_choices = (("mixed", "mixed", "mixed", "internal", "corner")
                                   if regime == 2 else
                                   ("internal", "edge", "corner", "mixed"))
                contact = contact_choices[int(rng.integers(len(contact_choices)))]
                if objects == 1 and contact == "mixed":
                    contact = "edge"
                required = depth_cap if regime == 2 or (regime == 1 and rng.random() < 0.45) else 0
                bar = regime == 2

            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, palette, objects, depth_cap,
                                  contact, required, bar)
                key = _grid_key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not make distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _extract_rectangles(grid, foreground):
    mask = grid == foreground
    seen = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    height, width = grid.shape
    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or seen[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            cells = []
            seen[start_row, start_col] = True
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells) + 1
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells) + 1
            assert len(cells) == (r1-r0)*(c1-c0)
            rectangles.append((r0, r1, c0, c1))
    assert rectangles
    for i, a in enumerate(rectangles):
        for b in rectangles[i+1:]:
            assert _separated(a, b)
    return sorted(rectangles)


def _perimeter(grid):
    return np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))


def _infer_conventions(train):
    background_candidates = set()
    for pair in train:
        edge_colors = np.unique(_perimeter(pair["input"]))
        if len(edge_colors) == 1:
            background_candidates.add(int(edge_colors[0]))
    assert len(background_candidates) == 1
    background = background_candidates.pop()
    input_colors = set(int(c) for c in np.unique(train[0]["input"]))
    assert len(input_colors) == 2 and background in input_colors
    foreground = (input_colors - {background}).pop()
    first = second = None
    shows_return = False
    max_train_depth = 0
    for pair in train:
        rectangles = _extract_rectangles(pair["input"], foreground)
        for r0, r1, c0, c1 in rectangles:
            depth = _depth((r0, r1, c0, c1))
            max_train_depth = max(max_train_depth, depth)
            if depth >= 1:
                value = int(pair["output"][r0+1, c0+1])
                assert first is None or first == value
                first = value
            if depth >= 2:
                value = int(pair["output"][r0+2, c0+2])
                assert second is None or second == value
                second = value
            if depth >= 3:
                assert int(pair["output"][r0+3, c0+3]) == first
                shows_return = True
    assert shows_return and max_train_depth >= 3
    assert len({background, foreground, first, second}) == 4
    return background, foreground, first, second, max_train_depth


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            assert not np.array_equal(pair["input"], pair["output"])
            assert _grid_key(pair["input"]) not in inputs
            assert _pair_key(pair) not in pairs
            inputs.add(_grid_key(pair["input"]))
            pairs.add(_pair_key(pair))

    background, foreground, accent1, accent2, train_depth = _infer_conventions(puzzle["train"])
    training_internal = training_mixed = training_deep_bar = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            assert set(int(c) for c in np.unique(inp)) == {background, foreground}
            rectangles = _extract_rectangles(inp, foreground)
            assert _signature(inp, background, foreground, rectangles) not in _SOURCE_INPUT_SIGNATURES
            assert np.all(out[inp == background] == background)
            assert set(int(c) for c in np.unique(out)).issubset(
                {background, foreground, accent1, accent2})
            assert max(_depth(rect) for rect in rectangles) <= train_depth

            # Recompute the rule from observed boxes and their four side distances.
            for r0, r1, c0, c1 in rectangles:
                rows = np.arange(r0, r1)[:, None]
                cols = np.arange(c0, c1)[None, :]
                distances = np.minimum(np.minimum(rows-r0, r1-1-rows),
                                       np.minimum(cols-c0, c1-1-cols))
                expected = np.where(distances == 0, foreground,
                                    np.where(distances % 2, accent1, accent2))
                assert np.array_equal(out[r0:r1, c0:c1], expected)

            if split == "train":
                height, width = inp.shape
                contacts = [(r0 == 0 or r1 == height or c0 == 0 or c1 == width)
                            for r0, r1, c0, c1 in rectangles]
                training_internal |= not any(contacts)
                training_mixed |= any(contacts) and not all(contacts)
                training_deep_bar |= (any(_depth(r) >= 3 for r in rectangles)
                                      and any(min(r[1]-r[0], r[3]-r[2]) <= 2 for r in rectangles))
    assert training_internal and training_mixed and training_deep_bar


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def _must_raise(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    a = generate_puzzle(314159)
    assert _puzzles_equal(a, generate_puzzle(314159))
    b = generate_puzzle(271828, 6, 3)
    assert _puzzles_equal(b, generate_puzzle(271828, 6, 3))
    assert not _puzzles_equal(b, generate_puzzle(271829, 6, 3))
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + 10*nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for seed, nt, ns in ((9101, None, 1), (9102, None, 4),
                         (9103, 2, None), (9104, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        _must_raise(lambda nt=nt, ns=ns: generate_puzzle(1, nt, ns))

    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    split_shapes = {"train": set(), "test": set()}
    object_counts = {"train": set(), "test": set()}
    boundary_cases = {"train": set(), "test": set()}
    depths, cores, kinds = set(), set(), set()
    varying_count_puzzles = 0
    train_corners = test_internal = comparable_pairs = 0
    overlap_rows = overlap_columns = combined_cases = deep_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        background, foreground, a1, a2, max_depth = _infer_conventions(puzzle["train"])
        palettes.add((background, foreground, a1, a2))
        if seed == 0:
            assert int(puzzle["train"][0]["input"][0, 0]) == foreground
        counts_here = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                rects = _extract_rectangles(grid, foreground)
                counts_here.add(len(rects))
                object_counts[split].add(len(rects))
                shapes.add(grid.shape)
                split_shapes[split].add(grid.shape)
                layouts.add((grid.shape, tuple(rects)))
                contacts = [(r0 == 0 or r1 == grid.shape[0] or c0 == 0 or c1 == grid.shape[1])
                            for r0, r1, c0, c1 in rects]
                boundary_cases[split].add("mixed" if any(contacts) and not all(contacts)
                                          else "edge" if any(contacts) else "internal")
                if split == "train":
                    train_corners += sum(
                        (r0 == 0 or r1 == grid.shape[0])
                        and (c0 == 0 or c1 == grid.shape[1])
                        for r0, r1, c0, c1 in rects
                    )
                else:
                    test_internal += not any(contacts)
                    deep_tests += any(_depth(r) >= 4 for r in rects)
                areas = [(r1-r0)*(c1-c0) for r0, r1, c0, c1 in rects]
                comparable_pairs += len(areas) >= 3 and max(areas) <= 1.7*min(areas)
                bar = any(min(r1-r0, c1-c0) <= 2 for r0, r1, c0, c1 in rects)
                deep = any(_depth(r) >= 3 for r in rects)
                combined_cases += bar and deep and any(contacts)
                for i, a in enumerate(rects):
                    for b in rects[i+1:]:
                        overlap_rows += max(a[0], b[0]) < min(a[1], b[1])
                        overlap_columns += max(a[2], b[2]) < min(a[3], b[3])
                for r in rects:
                    h, w = r[1]-r[0], r[3]-r[2]
                    d = _depth(r)
                    depths.add(d)
                    cores.add((h-2*d, w-2*d))
                    kinds.add("square" if h == w else "horizontal" if w > h else "vertical")
        varying_count_puzzles += len(counts_here) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert object_counts["train"] == {1, 2, 3, 4, 5}
    assert object_counts["test"] == {1, 2, 3, 4, 5}
    assert len(palettes) >= 25 and len(shapes) >= 80 and len(layouts) >= 200
    assert len(split_shapes["train"] & split_shapes["test"]) >= 10
    assert varying_count_puzzles >= 30
    assert {0, 1, 2, 3, 4}.issubset(depths)
    assert {"square", "horizontal", "vertical"}.issubset(kinds)
    assert {"internal", "edge", "mixed"}.issubset(boundary_cases["test"])
    assert train_corners >= 5 and test_internal >= 5
    assert comparable_pairs >= 10
    assert overlap_rows >= 40 and overlap_columns >= 40
    assert combined_cases >= 5 and deep_tests >= 1
    assert (1, 1) in cores and (2, 2) in cores
    assert any(min(h, w) <= 2 and max(h, w) >= 4 for h, w in cores)
