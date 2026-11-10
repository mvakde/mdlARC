import numpy as np


_SOURCE_TASK_ROWS = (
    (
        (
            "000000000",
            "444000000",
            "044000000",
            "444000000",
            "000004440",
            "000000400",
            "000000400",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "444000000",
            "744000000",
            "444000000",
            "000004440",
            "000007470",
            "000007470",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "444000000",
            "404000000",
            "004000000",
            "000004400",
            "000000440",
            "000004040",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "444000000",
            "474000000",
            "774000000",
            "000004470",
            "000007440",
            "000004740",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "000040000",
            "004400000",
            "004000000",
            "000000000",
            "000000000",
            "000004000",
            "000004440",
            "000000400",
        ),
        (
            "000000000",
            "007740000",
            "004470000",
            "004770000",
            "000000000",
            "000000000",
            "000004770",
            "000004440",
            "000007470",
        ),
    ),
)



def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (not _is_int(value) or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(test + 1, 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _connected(mask, diagonal=True):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in offsets:
            point = (r + dr, c + dc)
            if point in cells and point not in seen:
                seen.add(point)
                stack.append(point)
    return seen == cells


def _witnesses(mask):
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _boundary(r, c, h, w):
    return r in (0, h - 1) or c in (0, w - 1)


def _prune(mask, rng, target_missing, allowed):
    h, w = mask.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if allowed(r, c)]
    for _ in range(4):
        if mask.size - int(mask.sum()) >= target_missing:
            break
        for index in rng.permutation(len(candidates)):
            if mask.size - int(mask.sum()) >= target_missing:
                break
            r, c = candidates[int(index)]
            if not mask[r, c]:
                continue
            mask[r, c] = False
            if not _witnesses(mask) or not _connected(mask):
                mask[r, c] = True
    return mask


def _sample_mask(rng, h, w, style):
    if min(h, w) < 2 or h * w < 6:
        raise ValueError("box must allow an unambiguous partial rectangle")
    area = h * w
    mask = np.ones((h, w), dtype=bool)
    interior = lambda r, c: not _boundary(r, c, h, w)
    boundary = lambda r, c: _boundary(r, c, h, w)
    any_cell = lambda r, c: True

    if style == "interior":
        if min(h, w) < 3:
            raise ValueError("interior hole needs both box dimensions at least three")
        available = (h - 2) * (w - 2)
        missing = int(rng.integers(1, min(available, max(1, area // 3)) + 1))
        _prune(mask, rng, missing, interior)
    elif style == "boundary":
        missing = int(rng.integers(1, max(2, area // 3) + 1))
        _prune(mask, rng, missing, boundary)
    elif style == "near_complete":
        missing = int(rng.integers(1, min(3, area - 2)))
        _prune(mask, rng, missing, any_cell)
    elif style == "diagonal_spur":
        r = 0 if int(rng.integers(2)) == 0 else h - 1
        c = 0 if int(rng.integers(2)) == 0 else w - 1
        nr = 1 if r == 0 else h - 2
        nc = 1 if c == 0 else w - 2
        mask[r, nc] = False
        mask[nr, c] = False
        assert _connected(mask) and _witnesses(mask) and not _connected(mask, False)
        extra = int(rng.integers(0, max(2, area // 5)))
        allowed = boundary if int(rng.integers(3)) == 0 else any_cell
        _prune(mask, rng, 2 + extra, allowed)
    elif style == "mixed":
        if min(h, w) < 3:
            raise ValueError("mixed gap geometry needs an interior")
        ir = int(rng.integers(1, h - 1))
        ic = int(rng.integers(1, w - 1))
        mask[ir, ic] = False
        _prune(mask, rng, 2, boundary)
        assert (~mask[1:-1, 1:-1]).any()
        _prune(mask, rng, int(rng.integers(2, max(3, area // 2))), any_cell)
    elif style == "sparse":
        low = max(2, area // 2)
        high = min(area - 2, int(area * 0.72))
        target = int(rng.integers(low, max(low, high) + 1))
        _prune(mask, rng, target, any_cell)
    elif style == "dense":
        target = int(rng.integers(1, max(2, int(area * 0.45)) + 1))
        _prune(mask, rng, target, any_cell)
    else:
        raise ValueError("unknown mask style")

    assert int(mask.sum()) < area and _witnesses(mask) and _connected(mask)
    return mask


def _box(start, shape):
    r, c = start
    h, w = shape
    return r, r + h - 1, c, c + w - 1


def _separated(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _attempt_free_layout(rng, shapes, h, w, mode):
    order = list(rng.permutation(len(shapes)))
    starts = [None] * len(shapes)
    boxes = []
    for n, index in enumerate(order):
        bh, bw = shapes[index]
        if bh > h or bw > w:
            return None
        candidates = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)]
        if mode == "edge" and n == 0:
            candidates = [(r, c) for r, c in candidates if r == 0 or c == 0 or r + bh == h or c + bw == w]
        elif mode == "center" and n == 0:
            candidates.sort(key=lambda p: abs(2 * p[0] + bh - h) + abs(2 * p[1] + bw - w))
            candidates = candidates[:max(1, min(9, len(candidates)))]
        if not candidates:
            return None
        for candidate_index in rng.permutation(len(candidates)):
            start = candidates[int(candidate_index)]
            box = _box(start, shapes[index])
            if all(_separated(box, other) for other in boxes):
                starts[index] = start
                boxes.append(box)
                break
        if starts[index] is None:
            return None
    return starts


def _attempt_line_layout(rng, shapes, h, w, mode):
    order = list(rng.permutation(len(shapes)))
    if mode == "row":
        needed = sum(shapes[i][1] for i in order) + len(order) - 1
        tallest = max(shape[0] for shape in shapes)
        if needed > w or tallest > h:
            return None
        row = int(rng.integers(0, h - tallest + 1))
        left = int(rng.integers(0, w - needed + 1))
        extra = w - needed - left
        gaps = list(rng.multinomial(extra, np.ones(len(order)) / len(order)))
        starts = [None] * len(shapes)
        col = left
        for k, i in enumerate(order):
            starts[i] = (row, col)
            col += shapes[i][1] + 1 + gaps[k]
        return starts
    needed = sum(shapes[i][0] for i in order) + len(order) - 1
    widest = max(shape[1] for shape in shapes)
    if needed > h or widest > w:
        return None
    col = int(rng.integers(0, w - widest + 1))
    top = int(rng.integers(0, h - needed + 1))
    extra = h - needed - top
    gaps = list(rng.multinomial(extra, np.ones(len(order)) / len(order)))
    starts = [None] * len(shapes)
    row = top
    for k, i in enumerate(order):
        starts[i] = (row, col)
        row += shapes[i][0] + 1 + gaps[k]
    return starts


def _new_layout(rng, shapes):
    count = len(shapes)
    largest_h = max(s[0] for s in shapes)
    largest_w = max(s[1] for s in shapes)
    extra_low = (1, 4, 7, 10)[count - 1]
    extra_high = (5, 10, 13, 16)[count - 1] if int(rng.integers(4)) == 0 else 19
    h = min(30, largest_h + int(rng.integers(extra_low, extra_high)))
    w = min(30, largest_w + int(rng.integers(extra_low, extra_high)))
    modes = ("free", "row", "column", "edge", "center")
    mode = modes[int(rng.choice(len(modes), p=(.35, .18, .18, .18, .11)))]
    for _ in range(11):
        for _ in range(30):
            if mode in ("row", "column"):
                starts = _attempt_line_layout(rng, shapes, h, w, mode)
            else:
                starts = _attempt_free_layout(rng, shapes, h, w, mode)
            if starts is not None:
                return h, w, tuple(starts)
        h = min(30, h + 2)
        w = min(30, w + 2)
        mode = "free"
    raise ValueError("could not place separated boxes")


def _random_shape(rng):
    if int(rng.integers(4)) == 0:
        short = int(rng.integers(2, 4))
        long = int(rng.integers(5, 9))
        return (short, long) if int(rng.integers(2)) else (long, short)
    h = int(rng.integers(2, 7))
    w = int(rng.integers(2, 7))
    if h * w < 6:
        w = 3
    return h, w


def _random_style(rng, split, shape):
    if split == "train":
        styles = ("dense", "boundary", "interior", "mixed", "diagonal_spur", "sparse", "near_complete")
        weights = (.18, .13, .10, .15, .18, .16, .10)
    else:
        styles = ("dense", "boundary", "interior", "mixed", "diagonal_spur", "sparse", "near_complete")
        weights = (.14, .10, .08, .13, .23, .25, .07)
    style = styles[int(rng.choice(len(styles), p=weights))]
    if min(shape) < 3 and style in ("interior", "mixed"):
        style = "boundary" if int(rng.integers(2)) else "diagonal_spur"
    return style


def _random_spec(rng, split):
    count = int(rng.choice((1, 2, 3, 4), p=(.20, .38, .29, .13)))
    shapes = [_random_shape(rng) for _ in range(count)]
    gap_mode = int(rng.choice(3, p=(.18, .18, .64)))
    if gap_mode == 0:
        shapes = [(max(3, h), max(3, w)) for h, w in shapes]
        styles = ["interior"] * count
    elif gap_mode == 1:
        styles = ["boundary"] * count
    else:
        styles = [_random_style(rng, split, shape) for shape in shapes]
    if split == "test" and int(rng.integers(6)) == 0:
        styles = ["dense" if min(shape) < 3 else "interior" for shape in shapes]
    return shapes, styles


def _featured_specs(rng):
    anchor_shapes = [(3, 3), (3, 3)]
    anchor_styles = ["diagonal_spur", "boundary"]
    if int(rng.integers(2)):
        anchor_styles.reverse()
    small = (2, int(rng.integers(3, 5)))
    long = (3, int(rng.integers(6, 9)))
    medium = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    shapes = [small, long, medium]
    shapes = [(w, h) if int(rng.integers(2)) else (h, w) for h, w in shapes]
    styles = ["boundary", "sparse", "interior"]
    order = rng.permutation(3)
    coverage_shapes = [shapes[int(i)] for i in order]
    coverage_styles = [styles[int(i)] for i in order]
    return [(anchor_shapes, anchor_styles), (coverage_shapes, coverage_styles)]


def _assemble_pair(rng, roles, shapes, styles, geometry=None):
    background, source, completion = roles
    if geometry is None:
        h, w, starts = _new_layout(rng, shapes)
    else:
        h, w, starts, shapes = geometry
        styles = [_random_style(rng, "train", shape) for shape in shapes]
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    for start, shape, style in zip(starts, shapes, styles):
        r, c = start
        bh, bw = shape
        mask = _sample_mask(rng, bh, bw, style)
        inp[r:r + bh, c:c + bw][mask] = source
        out[r:r + bh, c:c + bw] = np.where(mask, source, completion)
    return {"input": inp, "output": out}, (h, w, starts, shapes)


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_signatures():
    return {((len(i), len(i[0]), tuple(int(x) for row in i for x in row)),
             (len(o), len(o[0]), tuple(int(x) for row in o for x in row)))
            for i, o in _SOURCE_TASK_ROWS}


def _source_copy(pair):
    i, o = pair["input"], pair["output"]
    key = ((i.shape[0], i.shape[1], tuple(int(x) for x in i.flat)),
           (o.shape[0], o.shape[1], tuple(int(x) for x in o.flat)))
    return key in _source_signatures()


def _components(grid, source):
    mask = grid == source
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                point = stack.pop()
                cells.append(point)
                pr, pc = point
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = pr + dr, pc + dc
                        if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                                and mask[nr, nc] and not seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            components.append(cells)
    return components


def _infer_roles(train):
    roles = None
    for pair in train:
        inp, out = pair["input"], pair["output"]
        changed = inp != out
        assert changed.any()
        before = np.unique(inp[changed])
        after = np.unique(out[changed])
        assert len(before) == len(after) == 1
        background, completion = int(before[0]), int(after[0])
        colors = set(int(v) for v in np.unique(inp))
        assert len(colors) == 2 and background in colors and completion not in colors
        source = next(iter(colors - {background}))
        current = (background, source, completion)
        assert len(set(current)) == 3
        if roles is None:
            roles = current
        assert current == roles
    assert roles is not None
    return roles


def _analyze(pair, roles):
    background, source, completion = roles
    inp, out = pair["input"], pair["output"]
    assert set(int(v) for v in np.unique(inp)) == {background, source}
    assert set(int(v) for v in np.unique(out)) == {background, source, completion}
    comps = _components(inp, source)
    assert 1 <= len(comps) <= 4
    boxes = []
    masks = []
    expected = inp.copy()
    for comp in comps:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        box = (min(rs), max(rs), min(cs), max(cs))
        r0, r1, c0, c1 = box
        mask = inp[r0:r1 + 1, c0:c1 + 1] == source
        assert min(mask.shape) >= 2 and mask.size >= 6
        assert int(mask.sum()) == len(comp)
        assert int(mask.sum()) < mask.size
        assert _witnesses(mask) and _connected(mask)
        for other in boxes:
            assert _separated(box, other)
        region = expected[r0:r1 + 1, c0:c1 + 1]
        expected[r0:r1 + 1, c0:c1 + 1] = np.where(region == background, completion, region)
        boxes.append(box)
        masks.append(mask.copy())
    assert np.array_equal(expected, out)
    return {"boxes": boxes, "masks": masks}


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            sig = _signature(inp)
            pair_sig = (sig, _signature(out))
            assert sig not in inputs and pair_sig not in pairs
            assert not _source_copy(pair)
            inputs.add(sig)
            pairs.add(pair_sig)
    roles = _infer_roles(puzzle["train"])
    analyses = {s: [_analyze(pair, roles) for pair in puzzle[s]] for s in ("train", "test")}
    train_masks = [mask for analysis in analyses["train"] for mask in analysis["masks"]]
    train_shapes = {mask.shape for mask in train_masks}
    assert len(train_shapes) >= 2
    assert any(not _connected(mask, False) for mask in train_masks)
    assert any((~mask[1:-1, 1:-1]).any() for mask in train_masks if min(mask.shape) >= 3)
    assert any((~mask[0]).any() or (~mask[-1]).any() or (~mask[:, 0]).any() or (~mask[:, -1]).any() for mask in train_masks)
    assert len({len(a["boxes"]) for a in analyses["train"]}) >= 2
    return roles, analyses


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    roles = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    result = {"train": [], "test": []}
    seen = set()
    train_geometries = []
    featured = _featured_specs(rng)
    if int(rng.integers(2)):
        featured.reverse()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(30):
                geometry = None
                if split == "train" and index < 2:
                    shapes, styles = featured[index]
                else:
                    shapes, styles = _random_spec(rng, split)
                    if train_geometries and int(rng.integers(5)) == 0:
                        geometry = train_geometries[int(rng.integers(len(train_geometries)))]
                pair, used_geometry = _assemble_pair(rng, roles, shapes, styles, geometry)
                signature = _signature(pair["input"])
                if signature in seen or _source_copy(pair):
                    continue
                seen.add(signature)
                result[split].append(pair)
                if split == "train":
                    train_geometries.append(used_geometry)
                break
            else:
                raise ValueError(f"could not create unique {split} pair")
    validate_generated_puzzle(result)
    return result


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(99173, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def self_test() -> None:
    first = generate_puzzle(424242, 7, 3)
    assert _puzzle_signature(first) == _puzzle_signature(generate_puzzle(424242, 7, 3))
    assert _puzzle_signature(first) != _puzzle_signature(generate_puzzle(424243, 7, 3))
    assert _puzzle_signature(generate_puzzle(12345)) == _puzzle_signature(generate_puzzle(12345))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        _expect_value_error(**kwargs)

    train_counts, test_counts, puzzles = set(), set(), set()
    palettes, canvas_shapes, box_shapes, object_counts = set(), set(), set(), set()
    mask_signatures, layouts = set(), set()
    gap_pairs = set()
    diagonal_splits = set()
    sparse_splits = set()
    mixed_size_splits = set()
    same_row = same_col = edge = center = near_complete = missing_corner = multi_hole = 0
    familiar_dense_test = 0
    layout_reuse = train_layout_reuse = 0
    close_fitting = spacious = rectangular = simple_two_boxes = branched_sparse = 0
    within_canvas = []
    within_box_shapes = []
    within_object_counts = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        roles, analyses = _validate(puzzle)
        palettes.add(roles)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_signature(puzzle))
        train_geometry = set()
        puzzle_canvases = set()
        puzzle_box_shapes = set()
        puzzle_object_counts = set()
        for split in ("train", "test"):
            for pair, analysis in zip(puzzle[split], analyses[split]):
                canvas_shapes.add(pair["input"].shape)
                puzzle_canvases.add(pair["input"].shape)
                object_counts.add(len(analysis["boxes"]))
                puzzle_object_counts.add(len(analysis["boxes"]))
                shapes = tuple(mask.shape for mask in analysis["masks"])
                box_shapes.update(shapes)
                puzzle_box_shapes.update(shapes)
                if len(set(shapes)) > 1:
                    mixed_size_splits.add(split)
                layout = (pair["input"].shape, tuple(sorted(analysis["boxes"])))
                layouts.add(layout)
                if split == "test" and layout in train_geometry:
                    layout_reuse += 1
                if split == "train" and layout in train_geometry:
                    train_layout_reuse += 1
                if split == "train":
                    train_geometry.add(layout)
                boxes = analysis["boxes"]
                if len(boxes) > 1:
                    same_row += len({b[0] for b in boxes}) == 1
                    same_col += len({b[2] for b in boxes}) == 1
                h, w = pair["input"].shape
                rectangular += h != w
                span_h = max(b[1] for b in boxes) - min(b[0] for b in boxes) + 1
                span_w = max(b[3] for b in boxes) - min(b[2] for b in boxes) + 1
                close_fitting += h - span_h <= 4 and w - span_w <= 4
                spacious += h - span_h >= 8 and w - span_w >= 8
                simple_two_boxes += len(boxes) == 2 and all(mask.shape == (3, 3) for mask in analysis["masks"])
                edge += any(b[0] == 0 or b[2] == 0 or b[1] == h - 1 or b[3] == w - 1 for b in boxes)
                center += any(b[0] > 0 and b[2] > 0 and b[1] < h - 1 and b[3] < w - 1 for b in boxes)
                all_interior = all_boundary = True
                dense_only = True
                for mask in analysis["masks"]:
                    mask_signatures.add((mask.shape, mask.tobytes()))
                    interior_gap = min(mask.shape) >= 3 and bool((~mask[1:-1, 1:-1]).any())
                    boundary_gap = bool((~mask[0]).any() or (~mask[-1]).any() or (~mask[:, 0]).any() or (~mask[:, -1]).any())
                    all_interior &= interior_gap and not boundary_gap
                    all_boundary &= boundary_gap and not interior_gap
                    near_complete += mask.size - int(mask.sum()) <= 2
                    missing_corner += not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1])
                    multi_hole += int((~mask[1:-1, 1:-1]).sum()) >= 2 if min(mask.shape) >= 3 else 0
                    if not _connected(mask, False):
                        diagonal_splits.add(split)
                    if int(mask.sum()) <= mask.size // 2:
                        sparse_splits.add(split)
                        dense_only = False
                        branch = any(
                            sum(0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1] and mask[r + dr, c + dc]
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc) >= 3
                            for r, c in zip(*np.nonzero(mask))
                        )
                        branched_sparse += branch
                if all_interior:
                    gap_pairs.add("interior")
                elif all_boundary:
                    gap_pairs.add("boundary")
                else:
                    gap_pairs.add("mixed")
                if split == "test" and dense_only:
                    familiar_dense_test += 1
        within_canvas.append(len(puzzle_canvases))
        within_box_shapes.append(len(puzzle_box_shapes))
        within_object_counts.append(len(puzzle_object_counts))
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(puzzles) >= 40
    assert len(palettes) >= 25 and len(canvas_shapes) >= 40 and len(box_shapes) >= 15
    assert object_counts == {1, 2, 3, 4} and len(mask_signatures) >= 150 and len(layouts) >= 150
    assert gap_pairs == {"interior", "boundary", "mixed"}
    assert diagonal_splits == sparse_splits == mixed_size_splits == {"train", "test"}
    assert same_row > 0 and same_col > 0 and edge > 0 and center > 0
    assert near_complete > 0 and missing_corner > 0 and multi_hole > 0
    assert familiar_dense_test > 0 and layout_reuse > 0 and train_layout_reuse > 0
    assert close_fitting > 0 and spacious > 0 and rectangular > 0
    assert simple_two_boxes > 0 and branched_sparse > 0
    assert all(n >= 2 for n in within_canvas)
    assert all(n >= 2 for n in within_box_shapes)
    assert all(n >= 2 for n in within_object_counts)
    assert any(h > w for h, w in box_shapes) and any(w > h for h, w in box_shapes)
